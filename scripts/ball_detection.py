"""
Ball tracking and detection module.
"""
import cv2
import numpy as np
import math
from scripts import utils


class BallTracker:
    """Tracks ball positions and history across frames."""

    def __init__(
        self,
        max_history=15,
        min_cum_displacement=30.0,
        max_dist=200,
        max_missed=7,
        max_interpolate_frames=15,
        dead_window_frames=30,
        dead_move_px=6.0,
    ):
        self.max_history = max_history
        self.min_cum_displacement = min_cum_displacement
        self.max_dist = max_dist
        self.max_missed = max_missed
        self.max_interpolate_frames = max_interpolate_frames
        self.dead_window_frames = dead_window_frames
        self.dead_move_px = dead_move_px
        self.tracks = []
        self.next_id = 0

    def cumulative_displacement(self, history):
        """Compute cumulative displacement along a track."""
        total = 0.0
        for i in range(1, len(history)):
            x0, y0 = history[i - 1][0], history[i - 1][1]
            x1, y1 = history[i][0], history[i][1]
            total += math.hypot(x1 - x0, y1 - y0)
        return total

    def update(self, current_detections, frame_idx):
        """Update tracks with new detections and handle missing frames."""
        matched_indices = []

        for t in self.tracks:
            if t["missed"] > self.max_missed:
                continue
            last_cx, last_cy = t["history"][-1][:2]
            best_dist = float("inf")
            best_idx = -1

            for i, det in enumerate(current_detections):
                if i in matched_indices:
                    continue
                dist = math.hypot(det[0] - last_cx, det[1] - last_cy)
                if dist < self.max_dist and dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx != -1:
                det = current_detections[best_idx]
                matched_indices.append(best_idx)
                t["history"].append(det)
                if len(t["history"]) > self.max_history:
                    t["history"].pop(0)
                t["missed"] = 0
                t["prev_real"] = t.get("last_real")
                t["prev_real_frame"] = t.get("last_real_frame")
                t["last_real"] = det
                t["last_real_frame"] = frame_idx
            else:
                t["missed"] += 1
                if t["missed"] <= self.max_interpolate_frames:
                    prev_real = t.get("prev_real")
                    last_real = t.get("last_real")
                    prev_frame = t.get("prev_real_frame")
                    last_frame = t.get("last_real_frame")
                    if prev_real is not None and last_real is not None and prev_frame is not None:
                        dt = max(1, last_frame - prev_frame)
                        vx = (last_real[0] - prev_real[0]) / dt
                        vy = (last_real[1] - prev_real[1]) / dt
                        steps = max(1, frame_idx - last_frame)
                        pred_cx = last_real[0] + (vx * steps)
                        pred_cy = last_real[1] + (vy * steps)
                        w = max(1.0, last_real[4] - last_real[2])
                        h = max(1.0, last_real[5] - last_real[3])
                        pred_x1 = pred_cx - (w / 2.0)
                        pred_y1 = pred_cy - (h / 2.0)
                        pred_x2 = pred_cx + (w / 2.0)
                        pred_y2 = pred_cy + (h / 2.0)
                        pred_area = w * h
                        t["history"].append(
                            (pred_cx, pred_cy, pred_x1, pred_y1, pred_x2, pred_y2, 0.0, pred_area)
                        )
                        if len(t["history"]) > self.max_history:
                            t["history"].pop(0)

        for i, det in enumerate(current_detections):
            if i not in matched_indices:
                self.tracks.append(
                    {
                        "id": self.next_id,
                        "history": [det],
                        "missed": 0,
                        "is_active": False,
                        "is_dead": False,
                        "prev_real": None,
                        "last_real": det,
                        "prev_real_frame": None,
                        "last_real_frame": frame_idx,
                    }
                )
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t["missed"] <= self.max_missed]

        for t in self.tracks:
            if len(t["history"]) < 2:
                t["is_active"] = False
                t["is_dead"] = False
            else:
                disp = self.cumulative_displacement(t["history"])
                net_disp = math.hypot(
                    t["history"][-1][0] - t["history"][0][0],
                    t["history"][-1][1] - t["history"][0][1],
                )
                tail = (
                    t["history"][-self.dead_window_frames :]
                    if len(t["history"]) > self.dead_window_frames
                    else t["history"]
                )
                recent_disp = self.cumulative_displacement(tail)
                t["is_dead"] = recent_disp < self.dead_move_px
                min_net = max(utils.BALL_MOVE_MIN_PX, self.min_cum_displacement * 0.25)
                t["is_active"] = (
                    disp >= self.min_cum_displacement
                    and net_disp >= min_net
                    and not t["is_dead"]
                )

    def draw_trails(self, frame, color=(0, 165, 255)):
        """Draw ball trails on frame."""
        for t in self.tracks:
            if t["is_active"] and len(t["history"]) > 1:
                pts = np.array([pt[:2] for pt in t["history"]], dtype=np.int32)
                cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=3)
                for pt in t["history"]:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, color, -1)

    def get_track(self, track_id):
        """Retrieve track by ID."""
        for t in self.tracks:
            if t["id"] == track_id:
                return t
        return None


def get_nearest_ball_any(ball_tracker, px, py):
    """Find nearest ball to a point."""
    best = None
    best_dist = float("inf")
    for t in ball_tracker.tracks:
        if not t.get("history"):
            continue
        bx, by = t["history"][-1][:2]
        d = math.hypot(px - bx, py - by)
        if d < best_dist:
            best = t
            best_dist = d
    return best, best_dist


def get_nearest_active_ball(ball_tracker, px, py):
    """Find nearest active ball to a point."""
    best = None
    best_dist = float("inf")
    for t in ball_tracker.tracks:
        if not t["is_active"] or not t["history"] or t.get("missed", 0) > 0:
            continue
        bx, by = t["history"][-1][:2]
        d = math.hypot(px - bx, py - by)
        if d < best_dist:
            best_dist = d
            best = t
    return best, best_dist


def get_primary_active_ball(ball_tracker):
    """Get the primary active ball (highest confidence)."""
    active = [
        t
        for t in ball_tracker.tracks
        if t["is_active"] and t["history"] and t.get("missed", 0) == 0
    ]
    if not active:
        return None
    return max(active, key=lambda t: t["history"][-1][6])


def ball_near_recent(track, anchor, radius, window_frames):
    """Check if ball was near anchor in recent window."""
    if track is None or anchor is None:
        return False
    hist = track.get("history", [])
    if not hist:
        return False
    tail = hist[-window_frames:] if len(hist) > window_frames else hist
    for pt in tail:
        if math.hypot(pt[0] - anchor[0], pt[1] - anchor[1]) <= radius:
            return True
    return False


def ball_heading_toward(track, anchor, window_frames, angle_deg, min_disp_px):
    """Check if ball is heading toward anchor."""
    if track is None or anchor is None:
        return False
    if not np.isfinite(anchor[0]) or not np.isfinite(anchor[1]):
        return False
    hist = track.get("history", [])
    if len(hist) < 2:
        return False
    tail = hist[-window_frames:] if len(hist) > window_frames else hist
    if len(tail) < 2:
        return False
    x0, y0 = tail[0][:2]
    x1, y1 = tail[-1][:2]
    if not np.isfinite(x0) or not np.isfinite(y0) or not np.isfinite(x1) or not np.isfinite(y1):
        return False
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if not np.isfinite(dist) or dist < min_disp_px:
        return False
    to_player = np.array([anchor[0] - x1, anchor[1] - y1], dtype=float)
    to_norm = np.linalg.norm(to_player)
    if not np.isfinite(to_norm) or to_norm < 1e-6:
        return False
    denom = dist * to_norm
    if not np.isfinite(denom) or denom <= 1e-12:
        return False
    dot = (dx * to_player[0] + dy * to_player[1]) / denom
    if not np.isfinite(dot):
        return False
    dot = float(np.clip(dot, -1.0, 1.0))
    angle = math.degrees(math.acos(dot))
    return angle <= angle_deg and dot > 0


def ball_moved_recently(ball_tracker, window_frames, min_disp_px):
    """Check if any ball moved recently."""
    for t in ball_tracker.tracks:
        if t.get("is_dead"):
            continue
        if utils.track_moved_recently(t, window_frames, min_disp_px):
            return True
    return False
