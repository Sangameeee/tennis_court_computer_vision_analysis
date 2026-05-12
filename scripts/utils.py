"""
Utility functions and constants for ball detection and shot classification.
"""
import cv2
import numpy as np
import os
import math
import shutil

# --- Colab Support ---
IN_COLAB = False
try:
    from google.colab import drive
    IN_COLAB = True
except Exception:
    drive = None


# --- Configuration Constants ---
PIXELS_PER_METER = 100.0
SERVE_NO_BALL_SEC = 1.0
OTHER_STATIC_SEC = 1.0
PLAYER_STATIC_RATIO = 0.05
BALL_TURN_ANGLE_DEG = 25.0
WRIST_CLOSE_RATIO = 0.0
WRIST_CLOSE_PX = 4.0
BALL_MOVE_MIN_PX = max(8.0, PIXELS_PER_METER * 0.05)
BALL_LIVE_MIN_PX = max(2.0, BALL_MOVE_MIN_PX * 0.25)
BALL_TOWARD_ANGLE_DEG = 35.0
WALK_WINDOW_SEC = 1.0
WALK_WRIST_VEL_PX = 4.0
WALK_SHOULDER_VEL_PX = 5.0
WALK_LEG_SIGN_CHANGES = 2
INTERP_MAX_SEC = 0.5
SHOT_CONFIRM_SEC = 0.5
DEAD_BALL_WINDOW_SEC = 1.0
DEAD_BALL_MOVE_PX = max(6.0, PIXELS_PER_METER * 0.03)
BALL_MAX_TURN_DEG = 120.0
BALL_MAX_STEP_MULT = 3.0
BALL_INTERP_CONF_MAX = 0.01
PLAYER_STATIC_PX = max(12.0, PIXELS_PER_METER * 0.1)
BALL_AWAY_EPS_PX = 1.0
SWING_WRIST_VEL_PX = 18.0
SWING_DIR_ANGLE_DEG = 60.0
BALL_NEAR_RADIUS_RATIO = 1.6
BALL_NEAR_MIN_PX = 40.0
BALL_AWAY_MIN_PX = 2.0
BALL_RELEVANCE_SEC = 1.0

# --- Environment Setup ---
def get_local_root():
    """Get the directory where this script is located."""
    try:
        # Check if we're in a regular Python script context
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        return os.getcwd()


def ensure_drive_mounted():
    """Mount Google Drive if in Colab."""
    if not IN_COLAB:
        return False
    if os.path.isdir("/content/drive/MyDrive"):
        return True
    try:
        drive.mount("/content/drive")
    except Exception:
        return False
    return os.path.isdir("/content/drive/MyDrive")


def resolve_save_dirs(drive_save_path):
    """Resolve output directories for video and metadata."""
    local_root = os.path.join(get_local_root(), "outputs")
    drive_available = ensure_drive_mounted()
    save_root = drive_save_path if drive_available else local_root
    os.makedirs(save_root, exist_ok=True)
    runtime_root = "/content" if os.path.isdir("/content") else local_root
    temp_root = os.path.join(runtime_root, "runtime_outputs")
    os.makedirs(temp_root, exist_ok=True)
    return save_root, temp_root, drive_available
PALETTE = [
    (255, 56, 56),
    (56, 255, 56),
    (56, 150, 255),
    (255, 178, 29),
    (255, 56, 200),
    (56, 255, 220),
    (180, 56, 255),
    (255, 140, 56),
]


# --- Boundary & Court Geometry ---
VALUES_DATA = {
    "sample_image.png": {
        "fileref": "",
        "size": 3434687,
        "filename": "sample_image.png",
        "base64_img_data": "",
        "file_attributes": {},
        "regions": {
            "0": {
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": [
                        79.48051948051949,
                        673.2467532467533,
                        1217.142857142857,
                        1901.2987012987014,
                        930.3896103896104,
                        79.48051948051949,
                    ],
                    "all_points_y": [
                        870.3896103896104,
                        45.97402597402598,
                        45.97402597402598,
                        871.948051948052,
                        968.5714285714287,
                        870.3896103896104,
                    ],
                },
                "region_attributes": {"label": "boundary"},
            }
        },
    }
}

x_coords = VALUES_DATA["sample_image.png"]["regions"]["0"]["shape_attributes"]["all_points_x"]
y_coords = VALUES_DATA["sample_image.png"]["regions"]["0"]["shape_attributes"]["all_points_y"]

POLYGON_POINTS_INT = np.array([[int(x), int(y)] for x, y in zip(x_coords, y_coords)], np.int32)
POLYGON_POINTS_FOR_DRAW_AND_TEST = POLYGON_POINTS_INT.reshape((-1, 1, 2))


def get_class_color(cls_id):
    """Return color for a given class ID."""
    return PALETTE[int(cls_id) % len(PALETTE)]


def all_corners_inside(box_xyxy, poly):
    """Check if all corners of a bounding box are inside a polygon."""
    x1, y1, x2, y2 = map(int, box_xyxy)
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    return all(cv2.pointPolygonTest(poly, corner, False) >= 0 for corner in corners)


def center_inside(center_xy, poly):
    """Check if a point (center) is inside a polygon."""
    cx, cy = center_xy
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0


def compute_court_axis(poly):
    """Compute longitudinal court axis from court boundary polygon."""
    pts = poly.reshape(-1, 2).astype(float)
    pts_sorted = sorted(pts, key=lambda p: p[1])
    if len(pts_sorted) < 4:
        return np.array([0.0, 1.0], dtype=float)
    top = np.array(pts_sorted[:2], dtype=float)
    bottom = np.array(pts_sorted[-2:], dtype=float)
    axis = bottom.mean(axis=0) - top.mean(axis=0)
    if np.linalg.norm(axis) < 1e-6:
        return np.array([0.0, 1.0], dtype=float)
    return axis


# --- Pose/Skeleton Constants ---
# YOLOv8-pose keypoints indices:
# 0 nose, 5 L-shoulder, 6 R-shoulder, 7 L-elbow, 8 R-elbow, 9 L-wrist, 10 R-wrist,
# 11 L-hip, 12 R-hip, 13 L-knee, 14 R-knee, 15 L-ankle, 16 R-ankle
SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (11, 12),
    (5, 11),
    (6, 12),
]


def draw_pose(frame, kpts_xy, kpts_conf=None, color=(0, 255, 255), conf_th=0.25):
    """Draw pose skeleton on frame."""
    def ok(i):
        return True if kpts_conf is None else (kpts_conf[i] >= conf_th)

    for a, b in SKELETON:
        if ok(a) and ok(b):
            ax, ay = map(int, kpts_xy[a])
            bx, by = map(int, kpts_xy[b])
            cv2.line(frame, (ax, ay), (bx, by), color, 2, cv2.LINE_AA)

    for i, (x, y) in enumerate(kpts_xy):
        if ok(i):
            cv2.circle(frame, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)


def get_keypoint(kpts_xy, kpts_conf, idx, conf_th=0.25):
    """Get keypoint if it passes confidence threshold."""
    if kpts_xy is None or kpts_conf is None:
        return None
    if kpts_conf[idx] < conf_th:
        return None
    return kpts_xy[idx]


def wrists_close_enough(left_wrist, right_wrist, ref_len=None, min_px=25.0, ratio=0.35):
    """Check if wrists are close together (backhand detection)."""
    if left_wrist is None or right_wrist is None:
        return False
    dist = math.hypot(left_wrist[0] - right_wrist[0], left_wrist[1] - right_wrist[1])
    if ratio <= 0 or ref_len is None or ref_len <= 0:
        thresh = min_px
    else:
        thresh = max(min_px, ref_len * ratio)
    return dist <= thresh


def get_player_anchor(kpts_xy, kpts_conf, conf_th=0.25):
    """Get player center anchor (shoulder mid or hip mid)."""
    ls = get_keypoint(kpts_xy, kpts_conf, 5, conf_th=conf_th)
    rs = get_keypoint(kpts_xy, kpts_conf, 6, conf_th=conf_th)
    if ls is not None and rs is not None:
        return ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
    lh = get_keypoint(kpts_xy, kpts_conf, 11, conf_th=conf_th)
    rh = get_keypoint(kpts_xy, kpts_conf, 12, conf_th=conf_th)
    if lh is not None and rh is not None:
        return ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
    return None


# --- Math/Analysis Utilities ---
def compute_ball_direction(track, window=5, min_points=3):
    """Compute ball direction vector from track history."""
    if track is None:
        return None
    hist = list(track["history"])
    if len(hist) < min_points:
        return None
    tail = hist[-window:] if len(hist) > window else hist
    x0, y0 = tail[0][:2]
    x1, y1 = tail[-1][:2]
    dx, dy = x1 - x0, y1 - y0
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    return np.array([dx, dy], dtype=float)


def angle_between(v1, v2):
    """Compute angle between two vectors in degrees."""
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-6:
        return 0.0
    cos_val = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return math.degrees(math.acos(cos_val))


def ball_direction_changed(track, angle_thresh=25.0, window=4):
    """Check if ball direction changed significantly."""
    if track is None:
        return False
    hist = list(track["history"])
    if len(hist) < window * 2:
        return False

    prev = hist[-2 * window : -window]
    curr = hist[-window:]

    def direction(points):
        if len(points) < 2:
            return None
        dx = points[-1][0] - points[0][0]
        dy = points[-1][1] - points[0][1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        return np.array([dx, dy], dtype=float)

    v1 = direction(prev)
    v2 = direction(curr)
    if v1 is None or v2 is None:
        return False
    return angle_between(v1, v2) >= angle_thresh


def estimate_ball_speed(track, fps, pixels_per_meter):
    """Estimate ball speed in m/s."""
    if track is None:
        return None
    hist = list(track["history"])
    if len(hist) < 2:
        return None
    x0, y0 = hist[-2][:2]
    x1, y1 = hist[-1][:2]
    speed_px_per_sec = math.hypot(x1 - x0, y1 - y0) * fps
    if pixels_per_meter is None or pixels_per_meter <= 0:
        return None
    speed = speed_px_per_sec / pixels_per_meter
    if not math.isfinite(speed):
        return None
    return round(speed, 2)


def safe_float(value):
    """Safely convert value to float."""
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def safe_int(value):
    """Safely convert value to int."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def classify_shot_direction(track, court_axis_vec):
    """Classify shot direction (down-the-line, cross-court, lob, unknown)."""
    direction_vec = compute_ball_direction(track)
    if direction_vec is None:
        return "unknown", None

    hist = list(track["history"])
    if len(hist) >= 4:
        area_recent = (hist[-1][7] + hist[-2][7]) / 2.0
        area_prev = (hist[-3][7] + hist[-4][7]) / 2.0
        if area_prev > 0 and area_recent < area_prev * 0.75 and direction_vec[1] < 0:
            return "lob", direction_vec

    axis = np.array(court_axis_vec, dtype=float)
    if np.linalg.norm(axis) < 1e-6:
        axis = np.array([0.0, 1.0], dtype=float)
    axis /= np.linalg.norm(axis)
    dir_norm = direction_vec / np.linalg.norm(direction_vec)
    angle = math.degrees(math.acos(np.clip(abs(np.dot(axis, dir_norm)), -1.0, 1.0)))
    if angle < 25.0:
        return "down-the-line", direction_vec
    return "cross-court", direction_vec


def get_vertical_direction_label(track):
    """Get vertical direction label (upside/downside)."""
    direction_vec = compute_ball_direction(track)
    if direction_vec is None:
        return None
    return "upside" if direction_vec[1] < 0 else "downside"


def track_moved_recently(track, window_frames, min_disp_px):
    """Check if track moved at least min_disp_px in recent window."""
    hist = track.get("history", [])
    if len(hist) < 2:
        return False
    tail = hist[-window_frames:] if len(hist) > window_frames else hist
    total = 0.0
    for i in range(1, len(tail)):
        total += math.hypot(tail[i][0] - tail[i - 1][0], tail[i][1] - tail[i - 1][1])
    return total >= min_disp_px


def should_replace_same_second(existing, candidate):
    """Determine if candidate shot should replace existing shot in same second."""
    if candidate.get("ball_present") and not existing.get("ball_present"):
        return True
    if not candidate.get("ball_present") and existing.get("ball_present"):
        return False
    if not candidate.get("ball_present"):
        return False
    es = existing.get("ball_speed_mps")
    cs = candidate.get("ball_speed_mps")
    if es is not None and cs is not None:
        return cs < es
    return False
