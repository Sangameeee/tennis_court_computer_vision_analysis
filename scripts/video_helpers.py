
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
import numpy as np


class VideoNotebookUtils:
    """Common utilities for ROI filtering and drawing overlays.

    The notebook uses two different processing approaches; both share the same
    core helpers (ROI strict-inside checks and drawing primitives). This class
    provides those helpers as static/class methods.
    """

    PALETTE: Sequence[Tuple[int, int, int]] = (
        (255, 56, 56),
        (56, 255, 56),
        (56, 150, 255),
        (255, 178, 29),
        (255, 56, 200),
        (56, 255, 220),
        (180, 56, 255),
        (255, 140, 56),
    )

    SIMPLE_SKELETON: Sequence[Tuple[int, int]] = (
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 6),
        (11, 12),
        (5, 11),
        (6, 12),
    )

    COCO_SKELETON_PAIRS: Sequence[Tuple[int, int]] = (
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    )

    @classmethod
    def coco_limb_colors(cls) -> Dict[Tuple[int, int], Tuple[int, int, int]]:
        """Return the default limb colors mapping for `COCO_SKELETON_PAIRS`."""

        return {
            (i, j): (255, 100, 0) if i < 6 or j < 6 else (0, 150, 255)
            for i, j in cls.COCO_SKELETON_PAIRS
        }

    @staticmethod
    def polygon_from_via(
        values_data: Mapping[str, Any],
        image_key: str,
        region_id: str = "0",
        *,
        reshape_for_opencv: bool = True,
    ) -> np.ndarray:
        """Build polygon points from a VIA-like annotation dict.

        Args:
            values_data: VIA-style JSON/dict containing polygon points.
            image_key: Key for the image entry inside `values_data`.
            region_id: Region id inside the VIA `regions` object.
            reshape_for_opencv: If True returns shape `(N, 1, 2)`.

        Returns:
            Numpy array of dtype `np.int32` with polygon vertices.
        """

        shape = values_data[image_key]["regions"][region_id]["shape_attributes"]
        x_coords = shape["all_points_x"]
        y_coords = shape["all_points_y"]

        pts = np.array([[int(x), int(y)] for x, y in zip(x_coords, y_coords)], np.int32)
        return pts.reshape((-1, 1, 2)) if reshape_for_opencv else pts

    @staticmethod
    def all_corners_inside(box_xyxy: Sequence[float], poly: np.ndarray) -> bool:
        """Return True only if all 4 bbox corners are inside `poly` (strict ROI)."""

        if cv2 is None:
            raise ModuleNotFoundError(
                "OpenCV is required for ROI checks. Install with `pip install opencv-python`."
            )

        x1, y1, x2, y2 = map(int, box_xyxy)
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        return all(cv2.pointPolygonTest(poly, corner, False) >= 0 for corner in corners)

    @classmethod
    def get_class_color(cls, cls_id: int) -> Tuple[int, int, int]:
        """Return a stable RGB color for a class id using `PALETTE`."""

        return cls.PALETTE[int(cls_id) % len(cls.PALETTE)]

    @classmethod
    def get_color(cls, cls_id: int) -> Tuple[int, int, int]:
        """Alias for `get_class_color` (kept for notebook compatibility)."""

        return cls.get_class_color(cls_id)

    @classmethod
    def draw_pose(
        cls,
        frame: np.ndarray,
        kpts_xy: np.ndarray,
        kpts_conf: Optional[np.ndarray] = None,
        *,
        color: Tuple[int, int, int] = (0, 255, 255),
        conf_th: float = 0.25,
        skeleton: Optional[Iterable[Tuple[int, int]]] = None,
    ) -> None:
        """Draw keypoints + skeleton lines on a frame.

        This matches the notebook's method-1 pose drawing behavior.
        """

        if cv2 is None:
            raise ModuleNotFoundError(
                "OpenCV is required for drawing. Install with `pip install opencv-python`."
            )

        if skeleton is None:
            skeleton = cls.SIMPLE_SKELETON

        def ok(i: int) -> bool:
            return True if kpts_conf is None else (kpts_conf[i] >= conf_th)

        for a, b in skeleton:
            if ok(a) and ok(b):
                ax, ay = map(int, kpts_xy[a])
                bx, by = map(int, kpts_xy[b])
                cv2.line(frame, (ax, ay), (bx, by), color, 2, cv2.LINE_AA)

        for i, (x, y) in enumerate(kpts_xy):
            if ok(i):
                cv2.circle(frame, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)

    @staticmethod
    def classify_shot_from_pose(
        kpts_xy: np.ndarray,
        kpts_conf: Optional[np.ndarray] = None,
        *,
        conf_th: float = 0.25,
    ) -> str:
        """Heuristic-only: returns 'serve' | 'forehand' | 'backhand' | 'unknown'."""

        def get(i: int) -> Optional[np.ndarray]:
            if kpts_conf is not None and kpts_conf[i] < conf_th:
                return None
            return kpts_xy[i]

        LS, RS = get(5), get(6)
        LW, RW = get(9), get(10)
        LE, RE = get(7), get(8)
        N = get(0)

        if LS is None or RS is None:
            return "unknown"

        shoulder_mid = (LS + RS) / 2.0
        shoulder_y = shoulder_mid[1]

        wrist = elbow = None
        if LW is not None and RW is not None:
            if abs(LW[0] - shoulder_mid[0]) >= abs(RW[0] - shoulder_mid[0]):
                wrist, elbow = LW, LE
            else:
                wrist, elbow = RW, RE
        elif LW is not None:
            wrist, elbow = LW, LE
        elif RW is not None:
            wrist, elbow = RW, RE
        else:
            return "unknown"

        head_y = N[1] if N is not None else (shoulder_y - 60)
        is_wrist_high = wrist[1] < (shoulder_y - 20)
        is_elbow_high = (elbow is not None) and (elbow[1] < (shoulder_y - 10))
        is_near_head = wrist[1] < (head_y + 30)
        if is_wrist_high and is_elbow_high and is_near_head:
            return "serve"

        crosses_midline = (wrist[0] - shoulder_mid[0]) * ((RS[0] - LS[0])) < 0
        return "backhand" if crosses_midline else "forehand"

    @staticmethod
    def draw_box(
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        label: str,
        conf: float,
        color: Tuple[int, int, int],
        *,
        thickness: int = 2,
    ) -> None:
        """Draw a labeled bounding box with a filled label background."""

        if cv2 is None:
            raise ModuleNotFoundError(
                "OpenCV is required for drawing. Install with `pip install opencv-python`."
            )

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        tag = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame,
            tag,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    @classmethod
    def draw_skeleton(
        cls,
        frame: np.ndarray,
        kps: np.ndarray,
        bbox_xyxy: Sequence[float],
        poly: np.ndarray,
        *,
        conf_thresh: float,
        skeleton_pairs: Optional[Iterable[Tuple[int, int]]] = None,
        limb_colors: Optional[Mapping[Tuple[int, int], Tuple[int, int, int]]] = None,
        kp_color: Tuple[int, int, int] = (0, 255, 180),
    ) -> None:
        """Draw a skeleton only when the bbox is strictly inside the ROI.

        This matches the notebook's method-2 skeleton drawing behavior.
        """

        if cv2 is None:
            raise ModuleNotFoundError(
                "OpenCV is required for drawing. Install with `pip install opencv-python`."
            )

        if not cls.all_corners_inside(bbox_xyxy, poly):
            return

        if skeleton_pairs is None:
            skeleton_pairs = cls.COCO_SKELETON_PAIRS
        if limb_colors is None:
            limb_colors = cls.coco_limb_colors()

        valid = {
            i: (int(x), int(y))
            for i, (x, y, c) in enumerate(kps)
            if c > conf_thresh and x > 0 and y > 0
        }
        if len(valid) < 3:
            return

        for i, j in skeleton_pairs:
            if i in valid and j in valid:
                color = limb_colors.get((i, j), (200, 200, 200))
                cv2.line(frame, valid[i], valid[j], color, 2, cv2.LINE_AA)

        for x, y in valid.values():
            cv2.circle(frame, (x, y), 4, kp_color, -1)
            cv2.circle(frame, (x, y), 4, (0, 0, 0), 1)
