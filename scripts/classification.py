"""
Real-Time Racket Sports Video Analysis Pipeline
================================================
Priority Features Implemented:
  1. Robust Serve Detection with personal boundary toss tracking
  2. Shot Candidate detection via skeleton wrist acceleration
  3. Shot Validation using 0.5-second ball-away rule
  4. Full integration with existing tracking & filtering modules

Usage:
  pipeline = RacketSportsPipeline(fps=30.0)
  for frame_idx, frame in enumerate(video_stream):
      results = pipeline.process_frame(frame_idx, detections, poses, ball_tracks)
      # results contains confirmed shots, serve states, and tracking metadata
"""

import json
import math
import os
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Assume these exist in your project structure
from scripts import ball_detection, utils

# ---------------------------------------------------------------------------
# Optional scipy import for Hungarian matching. Falls back to greedy if absent.
# ---------------------------------------------------------------------------
try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ===========================================================================
# EXISTING MODULES (Preserved exactly as provided)
# ===========================================================================

class BallDisappearanceBuffer:
    """Keeps the last known ball position alive during occlusion gaps."""
    def __init__(self, max_gap_frames: int = 12, max_extrap_frames: int = 6):
        self.max_gap_frames = max_gap_frames
        self.max_extrap_frames = max_extrap_frames
        self._state: dict = {}

    def update(self, ball_tracker, ball_id):
        if ball_id is None: return
        track = ball_tracker.get_track(ball_id) if ball_id is not None else None
        visible = (track is not None and track.get("history") and not track.get("is_dead"))
        st = self._state.setdefault(ball_id, {"pos": None, "vel": (0.0, 0.0), "gap": 0})
        if visible:
            hist = track["history"]
            bx, by = hist[-1][:2]
            if len(hist) >= 2:
                pbx, pby = hist[-2][:2]
                st["vel"] = (bx - pbx, by - pby)
            else:
                st["vel"] = (0.0, 0.0)
            st["pos"] = (bx, by)
            st["gap"] = 0
        else:
            st["gap"] += 1

    def get_position(self, ball_id):
        if ball_id is None: return None
        st = self._state.get(ball_id)
        if st is None or st["pos"] is None: return None
        if st["gap"] == 0: return st["pos"]
        if st["gap"] > self.max_gap_frames: return None
        extrap_steps = min(st["gap"], self.max_extrap_frames)
        vx, vy = st["vel"]
        px, py = st["pos"]
        decay = 0.85
        ex, ey = px, py
        cvx, cvy = vx, vy
        for _ in range(extrap_steps):
            cvx *= decay; cvy *= decay
            ex += cvx; ey += cvy
        return (ex, ey)

    def is_visible(self, ball_id) -> bool:
        st = self._state.get(ball_id)
        return st is not None and st.get("gap", 1) == 0

    def gap_frames(self, ball_id) -> int:
        st = self._state.get(ball_id)
        return st["gap"] if st else 0

    def reset(self, ball_id):
        self._state.pop(ball_id, None)


class GhostBallFilter:
    """Suppresses spurious ball detections that cluster in the mid-court region."""
    def __init__(self, court_centre_zone=(0.3, 0.35, 0.7, 0.65), min_static_frames: int = 8,
                 static_move_px: float = 6.0, min_history_to_trust: int = 10,
                 frame_w: int = 1920, frame_h: int = 1080):
        self.zone = court_centre_zone
        self.min_static_frames = min_static_frames
        self.static_move_px = static_move_px
        self.min_history_to_trust = min_history_to_trust
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._pos_history: dict = {}

    def _in_zone(self, x, y) -> bool:
        x0, y0, x1, y1 = self.zone
        if x0 <= 1.0 and y0 <= 1.0:
            fx, fy = x / self.frame_w, y / self.frame_h
            return x0 <= fx <= x1 and y0 <= fy <= y1
        return x0 <= x <= x1 and y0 <= y <= y1

    def update_and_filter(self, ball_tracker) -> set:
        suppressed = set()
        if ball_tracker is None: return suppressed
        active_tracks = [t for t in getattr(ball_tracker, "tracks", {}).values()
                         if not t.get("is_dead") and t.get("history")]
        for t in active_tracks:
            tid = t["id"]
            bx, by = t["history"][-1][:2]
            hist = self._pos_history.setdefault(tid, deque(maxlen=self.min_static_frames + 2))
            hist.append((bx, by))
            if len(t["history"]) >= self.min_history_to_trust: continue
            if not self._in_zone(bx, by): continue
            if len(hist) >= self.min_static_frames:
                xs = [p[0] for p in hist]; ys = [p[1] for p in hist]
                spread = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
                if spread <= self.static_move_px: suppressed.add(tid)
        return suppressed

    def best_ball(self, ball_tracker, anchor_x, anchor_y, suppressed=None):
        if ball_tracker is None: return None, None
        if suppressed is None: suppressed = self.update_and_filter(ball_tracker)
        active = [t for t in getattr(ball_tracker, "tracks", {}).values()
                  if not t.get("is_dead") and t.get("history")]
        best_real, best_real_d = None, float("inf")
        best_fallback, best_fallback_d = None, float("inf")
        for t in active:
            bx, by = t["history"][-1][:2]
            d = math.hypot(bx - anchor_x, by - anchor_y)
            if t["id"] not in suppressed:
                if d < best_real_d: best_real_d, best_real = d, t
            else:
                if len(t["history"]) >= self.min_history_to_trust:
                    if d < best_fallback_d: best_fallback_d, best_fallback = d, t
        return (best_real, best_real_d) if best_real else (best_fallback, best_fallback_d)


class SkeletonMovementTracker:
    """Tracks wrist positions per player to gate shot classification on swing motion."""
    def __init__(self, movement_thresh: float = 15.0, history_len: int = 5):
        self.history: dict = {}
        self.movement_thresh = movement_thresh
        self.history_len = history_len

    def update(self, player_id, left_wrist, right_wrist):
        if player_id is None: return
        hist = self.history.setdefault(player_id, deque(maxlen=self.history_len))
        hist.append((left_wrist, right_wrist))

    @staticmethod
    def _max_disp(seq):
        max_d = 0.0
        for i in range(1, len(seq)):
            prev_lw, prev_rw = seq[i - 1]
            cur_lw, cur_rw = seq[i]
            for prev, cur in ((prev_lw, cur_lw), (prev_rw, cur_rw)):
                if prev is None or cur is None: continue
                max_d = max(max_d, math.hypot(cur[0] - prev[0], cur[1] - prev[1]))
        return max_d

    def has_swing(self, player_id, window: int = None):
        hist = list(self.history.get(player_id, []))
        if len(hist) < 2: return False
        if window is not None: hist = hist[-max(2, window):]
        return self._max_disp(hist) >= self.movement_thresh

    def has_swing_recent(self, player_id, window: int = 3):
        return self.has_swing(player_id, window=window)

    def has_swing_now(self, player_id):
        hist = list(self.history.get(player_id, []))
        if len(hist) < 2: return False
        return self._max_disp(hist[-2:]) >= self.movement_thresh


class PoseHistoryTracker:
    """Stores recent pose keypoints per player for motion heuristics."""
    def __init__(self, history_len: int = 30, conf_th: float = 0.25):
        self.history: dict = {}
        self.history_len = history_len
        self.conf_th = conf_th

    def _kp(self, kpts_xy, kpts_conf, idx):
        if kpts_xy is None or kpts_conf is None or idx >= len(kpts_xy): return None
        if kpts_conf[idx] < self.conf_th: return None
        return kpts_xy[idx]

    def update(self, player_id, kpts_xy, kpts_conf):
        if player_id is None or kpts_xy is None or kpts_conf is None: return
        sample = {
            "lw": self._kp(kpts_xy, kpts_conf, 9), "rw": self._kp(kpts_xy, kpts_conf, 10),
            "ls": self._kp(kpts_xy, kpts_conf, 5), "rs": self._kp(kpts_xy, kpts_conf, 6),
            "lh": self._kp(kpts_xy, kpts_conf, 11), "rh": self._kp(kpts_xy, kpts_conf, 12),
            "lk": self._kp(kpts_xy, kpts_conf, 13), "rk": self._kp(kpts_xy, kpts_conf, 14),
            "la": self._kp(kpts_xy, kpts_conf, 15), "ra": self._kp(kpts_xy, kpts_conf, 16),
        }
        hist = self.history.setdefault(player_id, deque(maxlen=self.history_len))
        hist.append(sample)

    def wrist_speed_now(self, player_id):
        hist = self.history.get(player_id)
        if not hist or len(hist) < 2: return None
        prev, cur = hist[-2], hist[-1]
        max_disp = 0.0
        for key in ("lw", "rw"):
            p, c = prev.get(key), cur.get(key)
            if p is None or c is None: continue
            max_disp = max(max_disp, math.hypot(c[0] - p[0], c[1] - p[1]))
        return max_disp

    def shoulder_speed_now(self, player_id):
        hist = self.history.get(player_id)
        if not hist or len(hist) < 2: return None
        prev, cur = hist[-2], hist[-1]
        pls, prs = prev.get("ls"), prev.get("rs")
        cls, crs = cur.get("ls"), cur.get("rs")
        if pls is None or prs is None or cls is None or crs is None: return None
        prev_mid = ((pls[0] + prs[0]) / 2.0, (pls[1] + prs[1]) / 2.0)
        cur_mid = ((cls[0] + crs[0]) / 2.0, (cls[1] + crs[1]) / 2.0)
        return math.hypot(cur_mid[0] - prev_mid[0], cur_mid[1] - prev_mid[1])

    def leg_cycle_detected(self, player_id, window_frames: int, min_sign_changes: int = 2, min_ankle_diff_px: float = 8.0):
        hist = self.history.get(player_id)
        if not hist or len(hist) < 3: return False
        recent = list(hist)[-window_frames:]
        phases = []
        for sample in recent:
            la, ra = sample.get("la"), sample.get("ra")
            diff = None
            if la is not None and ra is not None: diff = la[1] - ra[1]
            else:
                lk, rk = sample.get("lk"), sample.get("rk")
                if lk is not None and rk is not None: diff = lk[1] - rk[1]
            if diff is not None and abs(diff) >= min_ankle_diff_px: phases.append(diff)
        if len(phases) < (min_sign_changes + 1): return False
        signs = [1 if p > 0 else -1 for p in phases]
        changes = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
        return changes >= min_sign_changes


class ShotCandidateTracker:
    """Tracks swing events and confirms shots once the ball moves away."""
    def __init__(self, confirm_frames: int, away_min_px: float, min_ball_move_px: float,
                 max_latch_dist: float = 300.0, wrist_revalidate_conf: float = 0.5):
        self.confirm_frames = confirm_frames
        self.away_min_px = away_min_px
        self.min_ball_move_px = min_ball_move_px
        self.max_latch_dist = max_latch_dist
        self.wrist_revalidate_conf = wrist_revalidate_conf
        self.candidates: dict = {}

    def add_candidate(self, player_id, frame_idx: int, anchor, ball_id, wrists_close: bool,
                      serve_context: bool, start_dist, lw=None, rw=None):
        if player_id is None or anchor is None: return
        self.candidates[player_id] = {
            "start_frame": frame_idx, "expires_frame": frame_idx + self.confirm_frames,
            "anchor": anchor, "ball_id": ball_id, "wrists_close": wrists_close,
            "serve_context": serve_context, "last_dist": start_dist,
            "lw": lw, "rw": rw, "wrist_confirmed": False,
        }

    def drop_candidate(self, player_id):
        self.candidates.pop(player_id, None)

    def update_wrists(self, player_id, lw, rw, lw_conf: float, rw_conf: float):
        cand = self.candidates.get(player_id)
        if cand is None or cand["wrist_confirmed"]: return
        if lw_conf >= self.wrist_revalidate_conf and rw_conf >= self.wrist_revalidate_conf:
            if lw is not None and rw is not None:
                dist = math.hypot(lw[0] - rw[0], lw[1] - rw[1])
                cand["wrists_close"] = dist < 60.0
                cand["wrist_confirmed"] = True
                cand["lw"], cand["rw"] = lw, rw

    def update(self, frame_idx: int, ball_tracker, anchors_by_player: dict,
               blocked_players=None, suppressed_ball_ids: set = None,
               disappearance_buffer: BallDisappearanceBuffer = None):
        if suppressed_ball_ids is None: suppressed_ball_ids = set()
        confirmed = []
        for player_id, cand in list(self.candidates.items()):
            if blocked_players and player_id in blocked_players: continue
            if frame_idx > cand["expires_frame"]:
                del self.candidates[player_id]; continue
            anchor = anchors_by_player.get(player_id) or cand["anchor"]
            if anchor is None: continue

            if cand["ball_id"] is None or cand["ball_id"] in suppressed_ball_ids:
                nearest_track, nearest_dist = ball_detection.get_nearest_ball_any(ball_tracker, anchor[0], anchor[1])
                if (nearest_track is not None and nearest_dist < self.max_latch_dist
                        and nearest_track["id"] not in suppressed_ball_ids):
                    cand["ball_id"] = nearest_track["id"]
                else:
                    continue

            pos = None
            if disappearance_buffer is not None:
                pos = disappearance_buffer.get_position(cand["ball_id"])
            if pos is None:
                track = ball_tracker.get_track(cand["ball_id"])
                if track is None or not track.get("history") or track.get("is_dead"): continue
                pos = track["history"][-1][:2]

            bx, by = pos
            dist = math.hypot(bx - anchor[0], by - anchor[1])
            if cand["last_dist"] is None:
                cand["last_dist"] = dist

            step = 0.0
            if disappearance_buffer is not None and not disappearance_buffer.is_visible(cand["ball_id"]):
                vx, vy = disappearance_buffer._state.get(cand["ball_id"], {}).get("vel", (0.0, 0.0))
                gap = disappearance_buffer.gap_frames(cand["ball_id"])
                decay = 0.85 ** min(gap, disappearance_buffer.max_extrap_frames)
                step = math.hypot(vx * decay, vy * decay)
            else:
                track = ball_tracker.get_track(cand["ball_id"])
                if track and track.get("history") and len(track["history"]) >= 2:
                    pbx, pby = track["history"][-2][:2]
                    step = math.hypot(bx - pbx, by - pby)

            if step >= self.away_min_px and dist >= cand["last_dist"] + self.away_min_px:
                shot_label = "serve" if cand["serve_context"] else ("backhand" if cand["wrists_close"] else "forehand")
                confirmed.append({"player_id": player_id, "shot": shot_label, "ball_id": cand["ball_id"]})
                del self.candidates[player_id]
            else:
                cand["last_dist"] = dist
        return confirmed


class PersonTracker:
    """Tracks person/player centres and bounding boxes across frames."""
    def __init__(self, track_file=None, max_missed: int = 30, max_dist: float = 150.0):
        self.tracks: list = []
        self.next_id: int = 0
        self.max_missed = max_missed
        self.max_dist = max_dist
        self.track_file = track_file
        if track_file is not None and os.path.exists(track_file):
            try:
                with open(track_file, "r") as f: data = json.load(f)
                for t in data.get("tracks", []):
                    self.tracks.append({"id": t["id"], "center": tuple(t["center"]),
                                        "bbox": tuple(t.get("bbox", (0, 0, 0, 0))), "missed": 0})
                self.next_id = data.get("next_id", self.next_id)
            except Exception: pass

    def save(self):
        if not self.track_file: return
        data = {"next_id": self.next_id, "tracks": [{"id": t["id"], "center": list(t["center"]),
                "bbox": list(t.get("bbox", (0, 0, 0, 0)))} for t in self.tracks]}
        try:
            with open(self.track_file, "w") as f: json.dump(data, f)
        except Exception: pass

    def update(self, detections):
        if not self.tracks or not detections:
            for det in detections:
                self.tracks.append({"id": self.next_id, "center": (det[0], det[1]),
                                    "bbox": (det[2], det[3], det[4], det[5]), "missed": 0})
                self.next_id += 1
            if not detections:
                for t in self.tracks: t["missed"] += 1
            self.tracks = [t for t in self.tracks if t["missed"] <= self.max_missed]
            return

        matched_track_indices = set()
        matched_det_indices = set()
        if _SCIPY_AVAILABLE and len(self.tracks) > 1 and len(detections) > 1:
            cost = np.array([[math.hypot(det[0] - t["center"][0], det[1] - t["center"][1])
                              for det in detections] for t in self.tracks], dtype=float)
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < self.max_dist:
                    det = detections[c]
                    self.tracks[r]["center"] = (det[0], det[1])
                    self.tracks[r]["bbox"] = (det[2], det[3], det[4], det[5])
                    self.tracks[r]["missed"] = 0
                    matched_track_indices.add(r); matched_det_indices.add(c)
        else:
            for ti, t in enumerate(self.tracks):
                best_i, best_d = -1, float("inf")
                for di, det in enumerate(detections):
                    if di in matched_det_indices: continue
                    d = math.hypot(det[0] - t["center"][0], det[1] - t["center"][1])
                    if d < best_d: best_d, best_i = d, di
                if best_i != -1 and best_d < self.max_dist:
                    det = detections[best_i]
                    matched_det_indices.add(best_i); matched_track_indices.add(ti)
                    t["center"] = (det[0], det[1]); t["bbox"] = (det[2], det[3], det[4], det[5]); t["missed"] = 0

        for ti, t in enumerate(self.tracks):
            if ti not in matched_track_indices: t["missed"] += 1
        for di, det in enumerate(detections):
            if di not in matched_det_indices:
                self.tracks.append({"id": self.next_id, "center": (det[0], det[1]),
                                    "bbox": (det[2], det[3], det[4], det[5]), "missed": 0})
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t["missed"] <= self.max_missed]

    def lookup_id_by_center(self, cx, cy, max_dist=None):
        md = self.max_dist if max_dist is None else max_dist
        best, best_d = None, float("inf")
        for t in self.tracks:
            d = math.hypot(cx - t["center"][0], cy - t["center"][1])
            if d < best_d and d < md: best_d, best = d, t["id"]
        return best


class PersonMotionTracker:
    """Tracks player motion to detect walking vs. stationary states."""
    def __init__(self, history_len: int = 45):
        self.history: dict = {}
        self.history_len = history_len

    def update(self, player_id, center, size=None):
        if player_id is None or center is None: return
        cx, cy = center
        w = max(1.0, float(size)) if size is not None else 1.0
        h = w
        hist = self.history.setdefault(player_id, deque(maxlen=self.history_len))
        hist.append((cx, cy, w, h))

    def is_static(self, player_id, window_frames: int, move_ratio: float = 0.05):
        hist = self.history.get(player_id)
        if hist is None or len(hist) < window_frames: return False
        recent = list(hist)[-window_frames:]
        x0, y0, w0, h0 = recent[0]
        base_size = max(w0, h0)
        max_disp = 0.0
        for cx, cy, w, h in recent:
            max_disp = max(max_disp, math.hypot(cx - x0, cy - y0))
            base_size = max(base_size, max(w, h))
        return max_disp <= (base_size * move_ratio)

    def others_static(self, current_id, window_frames: int, move_ratio: float = 0.05):
        other_ids = [pid for pid in self.history if pid != current_id]
        if not other_ids: return True
        return all(self.is_static(pid, window_frames, move_ratio=move_ratio) for pid in other_ids)

    def all_static(self, window_frames: int, move_ratio: float = 0.05):
        if not self.history: return False
        return all(self.is_static(pid, window_frames, move_ratio=move_ratio) for pid in self.history)


class MissedShotDetector:
    """Marks a shot as missed if the ball direction does not change within a timeout."""
    def __init__(self, fps: float, timeout_sec: float = 2.0, angle_thresh: float = 20.0, max_gap_frames: int = 15):
        self.pending: list = []
        self.fps = fps
        self.timeout_frames = int(fps * timeout_sec)
        self.angle_thresh = angle_thresh
        self.max_gap_frames = max_gap_frames

    def register_shot(self, shot_index: int, frame_idx: int, ball_id, initial_dir):
        if ball_id is None or initial_dir is None: return
        self.pending.append({"shot_index": shot_index, "frame_idx": frame_idx, "ball_id": ball_id,
                             "initial_dir": initial_dir, "last_dir": initial_dir, "invisible_frames": 0})

    def update_shot(self, shot_index: int, frame_idx: int, ball_id, initial_dir):
        if ball_id is None or initial_dir is None: return
        for p in self.pending:
            if p["shot_index"] == shot_index:
                p["ball_id"] = ball_id; p["last_dir"] = initial_dir; p["frame_idx"] = frame_idx; return
        self.register_shot(shot_index, frame_idx, ball_id, initial_dir)

    def update(self, frame_idx: int, ball_tracker, shots: list):
        remaining = []
        for p in self.pending:
            if frame_idx - p["frame_idx"] >= self.timeout_frames:
                shots[p["shot_index"]]["status"] = "missed"; shots[p["shot_index"]]["missed_frame"] = frame_idx; continue
            track = ball_tracker.get_track(p["ball_id"])
            current_dir = utils.compute_ball_direction(track)
            if current_dir is None:
                p["invisible_frames"] = p.get("invisible_frames", 0) + 1
                if p["invisible_frames"] >= self.max_gap_frames:
                    shots[p["shot_index"]]["status"] = "missed"; shots[p["shot_index"]]["missed_frame"] = frame_idx; continue
                remaining.append(p); continue
            p["invisible_frames"] = 0
            angle = utils.angle_between(p["last_dir"], current_dir)
            if angle >= self.angle_thresh: continue
            p["last_dir"] = current_dir
            remaining.append(p)
        self.pending = remaining


# ===========================================================================
# NEW: ServeDetectionTracker (Priority Feature)
# ===========================================================================

class ServeDetectionTracker:
    """
    Implements robust serve preparation & execution detection.
    
    State Machine per potential server:
      IDLE -> PREP -> READY -> EXECUTED -> IDLE
      
    Criteria:
      1. All players static for >= static_window_sec
      2. Ball stays within personal boundary (shoulder/torso center) for >= toss_window_sec
      3. Tolerates brief ball exits (toss & catch, detection jitter)
      4. Blocks large horizontal displacement away from server during prep
      5. READY state triggers when 1 & 2 are satisfied
      6. EXECUTED triggers on sharp wrist swing -> feeds ShotCandidateTracker
    """
    def __init__(self, fps: float, static_window_sec: float = 1.0, toss_window_sec: float = 1.2,
                 boundary_radius_px: float = 180.0, max_exit_frames: int = 8,
                 prep_timeout_sec: float = 4.0, horizontal_drift_thresh_px: float = 40.0):
        self.fps = fps
        self.static_frames = int(fps * static_window_sec)
        self.toss_frames = int(fps * toss_window_sec)
        self.boundary_radius = boundary_radius_px
        self.max_exit_frames = max_exit_frames
        self.timeout_frames = int(fps * prep_timeout_sec)
        self.drift_thresh = horizontal_drift_thresh_px

        # State tracking
        self.server_id: Optional[int] = None
        self.state: str = "IDLE"  # IDLE, PREP, READY, EXECUTED
        self.prep_start_frame: Optional[int] = None
        self.ready_frame: Optional[int] = None
        self.toss_inside_count: int = 0
        self.toss_exit_count: int = 0
        self.boundary_center: Optional[Tuple[float, float]] = None
        self.ball_x_history: deque = deque(maxlen=int(fps * 0.5))  # For drift check

    def _get_torso_center(self, pose_tracker: PoseHistoryTracker, pid: int) -> Optional[Tuple[float, float]]:
        """Returns shoulder midpoint, fallback to hip midpoint."""
        hist = pose_tracker.history.get(pid)
        if not hist: return None
        sample = hist[-1]
        ls, rs = sample.get("ls"), sample.get("rs")
        if ls and rs:
            return ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
        lh, rh = sample.get("lh"), sample.get("rh")
        if lh and rh:
            return ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
        return None

    def reset(self):
        self.server_id = None
        self.state = "IDLE"
        self.prep_start_frame = None
        self.ready_frame = None
        self.toss_inside_count = 0
        self.toss_exit_count = 0
        self.boundary_center = None
        self.ball_x_history.clear()

    def update(self, frame_idx: int, person_motion: PersonMotionTracker,
               pose_tracker: PoseHistoryTracker, ball_buffer: BallDisappearanceBuffer,
               ball_tracker, active_player_ids: List[int]) -> Dict[str, Any]:
        """
        Returns dict with:
          - "state": current serve state
          - "server_id": active server or None
          - "trigger_execute": True when swing detected in READY state
          - "boundary_center": (x,y) of server's personal zone
        """
        result = {"state": self.state, "server_id": self.server_id, 
                  "trigger_execute": False, "boundary_center": self.boundary_center}

        # 1. Global static check
        all_static = person_motion.all_static(self.static_frames)

        # 2. Get best ball position (visible or interpolated)
        ball_pos = None
        ball_id = None
        for bid in ball_buffer._state:
            pos = ball_buffer.get_position(bid)
            if pos:
                ball_pos = pos
                ball_id = bid
                break

        # 3. State Machine Transitions
        if self.state == "IDLE":
            if all_static and ball_pos:
                # Find player closest to ball as potential server
                best_pid, best_dist = None, float("inf")
                for pid in active_player_ids:
                    center = self._get_torso_center(pose_tracker, pid)
                    if center:
                        d = math.hypot(ball_pos[0] - center[0], ball_pos[1] - center[1])
                        if d < best_dist:
                            best_dist, best_pid = d, pid
                if best_pid and best_dist < self.boundary_radius * 1.5:
                    self.server_id = best_pid
                    self.state = "PREP"
                    self.prep_start_frame = frame_idx
                    self.boundary_center = self._get_torso_center(pose_tracker, best_pid)
                    self.toss_inside_count = 0
                    self.toss_exit_count = 0
                    self.ball_x_history.clear()

        elif self.state in ("PREP", "READY"):
            # Timeout guard
            if self.prep_start_frame and (frame_idx - self.prep_start_frame) > self.timeout_frames:
                self.reset()
                return result

            # Update boundary center dynamically (follows torso)
            if self.server_id:
                self.boundary_center = self._get_torso_center(pose_tracker, self.server_id)

            if self.boundary_center and ball_pos:
                dist = math.hypot(ball_pos[0] - self.boundary_center[0], ball_pos[1] - self.boundary_center[1])
                
                # Track horizontal drift to catch early serves/throws
                self.ball_x_history.append(ball_pos[0])
                if len(self.ball_x_history) >= 2:
                    x_drift = abs(self.ball_x_history[-1] - self.ball_x_history[0])
                    if x_drift > self.drift_thresh and dist > self.boundary_radius * 0.8:
                        self.reset()
                        return result

                # Toss boundary logic with exit tolerance
                if dist <= self.boundary_radius:
                    self.toss_inside_count += 1
                    self.toss_exit_count = max(0, self.toss_exit_count - 1)  # Decay exits on re-entry
                else:
                    self.toss_exit_count += 1
                    if self.toss_exit_count > self.max_exit_frames:
                        self.reset()
                        return result

                # Transition PREP -> READY
                if self.state == "PREP" and self.toss_inside_count >= self.toss_frames:
                    self.state = "READY"
                    self.ready_frame = frame_idx

                # If players start moving significantly during READY, abort
                if self.state == "READY" and not all_static:
                    if self.ready_frame and (frame_idx - self.ready_frame) > int(self.fps * 0.5):
                        self.reset()
                        return result

        result["state"] = self.state
        result["server_id"] = self.server_id
        result["boundary_center"] = self.boundary_center
        return result

    def check_execution_trigger(self, skeleton_tracker: SkeletonMovementTracker, frame_idx: int) -> bool:
        """Call when state == READY. Returns True if sharp wrist swing detected."""
        if self.state != "READY" or self.server_id is None:
            return False
        if skeleton_tracker.has_swing_recent(self.server_id, window=3):
            self.state = "EXECUTED"
            return True
        return False


# ===========================================================================
# INTEGRATION PIPELINE
# ===========================================================================

class RacketSportsPipeline:
    """
    Unified frame processor wiring all trackers together.
    Handles serve detection, shot candidates, validation, and missed shots.
    """
    def __init__(self, fps: float = 30.0, frame_w: int = 1920, frame_h: int = 1080):
        self.fps = fps
        self.frame_w = frame_w
        self.frame_h = frame_h

        # Initialize all trackers
        self.person_tracker = PersonTracker(max_missed=int(fps * 1.0), max_dist=150.0)
        self.person_motion = PersonMotionTracker(history_len=int(fps * 1.5))
        self.pose_tracker = PoseHistoryTracker(history_len=int(fps * 1.0))
        self.skeleton_tracker = SkeletonMovementTracker(movement_thresh=18.0, history_len=5)
        
        self.ball_buffer = BallDisappearanceBuffer(max_gap_frames=int(fps * 0.4), max_extrap_frames=int(fps * 0.2))
        self.ghost_filter = GhostBallFilter(frame_w=frame_w, frame_h=frame_h)
        
        # 0.5 second validation window as requested
        confirm_frames = int(fps * 0.5)
        self.shot_tracker = ShotCandidateTracker(
            confirm_frames=confirm_frames, away_min_px=12.0, min_ball_move_px=8.0, max_latch_dist=350.0
        )
        
        self.serve_tracker = ServeDetectionTracker(
            fps=fps, static_window_sec=1.0, toss_window_sec=1.2, 
            boundary_radius_px=180.0, max_exit_frames=8, prep_timeout_sec=4.0
        )
        
        self.missed_detector = MissedShotDetector(fps=fps, timeout_sec=2.0, angle_thresh=20.0)
        
        self.frame_idx = 0
        self.shots_log = []

    def process_frame(self, frame_idx: int, person_detections: list, 
                      pose_data: dict, ball_tracker) -> dict:
        """
        Args:
            frame_idx: Current frame number
            person_detections: List of (cx, cy, x1, y1, x2, y2)
            pose_data: Dict {player_id: {"kpts_xy": [...], "kpts_conf": [...]}}
            ball_tracker: Your existing ball tracking object with .tracks and .get_track()
        Returns:
            Dict with confirmed shots, serve state, and debug metadata
        """
        self.frame_idx = frame_idx
        output = {"confirmed_shots": [], "serve_state": "IDLE", "server_id": None, "debug": {}}

        # 1. Update Person & Motion Trackers
        self.person_tracker.update(person_detections)
        active_pids = [t["id"] for t in self.person_tracker.tracks]
        
        for t in self.person_tracker.tracks:
            pid = t["id"]
            cx, cy = t["center"]
            w = t["bbox"][2] - t["bbox"][0]
            self.person_motion.update(pid, (cx, cy), size=w)

        # 2. Update Pose & Skeleton Trackers
        for pid, pdata in pose_data.items():
            kpts_xy = pdata.get("kpts_xy")
            kpts_conf = pdata.get("kpts_conf")
            self.pose_tracker.update(pid, kpts_xy, kpts_conf)
            
            lw = self.pose_tracker.history[pid][-1].get("lw") if pid in self.pose_tracker.history else None
            rw = self.pose_tracker.history[pid][-1].get("rw") if pid in self.pose_tracker.history else None
            self.skeleton_tracker.update(pid, lw, rw)

        # 3. Update Ball Buffer & Ghost Filter
        active_ball_ids = [t["id"] for t in getattr(ball_tracker, "tracks", {}).values() if not t.get("is_dead")]
        for bid in active_ball_ids:
            self.ball_buffer.update(ball_tracker, bid)
            
        suppressed_balls = self.ghost_filter.update_and_filter(ball_tracker)

        # 4. Serve Detection Logic (Priority)
        serve_info = self.serve_tracker.update(
            frame_idx, self.person_motion, self.pose_tracker, 
            self.ball_buffer, ball_tracker, active_pids
        )
        output["serve_state"] = serve_info["state"]
        output["server_id"] = serve_info["server_id"]

        # Check for serve execution trigger
        if serve_info["state"] == "READY":
            if self.serve_tracker.check_execution_trigger(self.skeleton_tracker, frame_idx):
                server_id = self.serve_tracker.server_id
                anchor = self.pose_tracker._get_torso_center(self.pose_tracker, server_id) or \
                         next((t["center"] for t in self.person_tracker.tracks if t["id"] == server_id), None)
                if anchor:
                    # Feed directly into shot candidate tracker with serve_context=True
                    self.shot_tracker.add_candidate(
                        player_id=server_id, frame_idx=frame_idx, anchor=anchor,
                        ball_id=None, wrists_close=False, serve_context=True, start_dist=None
                    )
                self.serve_tracker.reset()

        # 5. Regular Shot Candidate Detection (Non-Serve)
        if serve_info["state"] == "IDLE":
            for pid in active_pids:
                if self.skeleton_tracker.has_swing_recent(pid, window=3):
                    # Avoid duplicate candidates
                    if pid not in self.shot_tracker.candidates:
                        anchor = next((t["center"] for t in self.person_tracker.tracks if t["id"] == pid), None)
                        if anchor:
                            lw = self.pose_tracker.history[pid][-1].get("lw")
                            rw = self.pose_tracker.history[pid][-1].get("rw")
                            wrists_close = (lw and rw and math.hypot(lw[0]-rw[0], lw[1]-rw[1]) < 60.0)
                            self.shot_tracker.add_candidate(
                                player_id=pid, frame_idx=frame_idx, anchor=anchor,
                                ball_id=None, wrists_close=wrists_close, serve_context=False, start_dist=None,
                                lw=lw, rw=rw
                            )

        # 6. Update Wrist Re-validation & Shot Confirmation
        anchors_map = {t["id"]: t["center"] for t in self.person_tracker.tracks}
        for pid in self.shot_tracker.candidates:
            if pid in pose_data:
                lw = self.pose_tracker.history[pid][-1].get("lw")
                rw = self.pose_tracker.history[pid][-1].get("rw")
                # Mock confidence extraction; replace with actual pose model confidences
                self.shot_tracker.update_wrists(pid, lw, rw, lw_conf=0.8, rw_conf=0.8)

        confirmed = self.shot_tracker.update(
            frame_idx, ball_tracker, anchors_map, 
            suppressed_ball_ids=suppressed_balls, disappearance_buffer=self.ball_buffer
        )
        
        for shot in confirmed:
            shot["frame"] = frame_idx
            shot["status"] = "pending_return"
            self.shots_log.append(shot)
            output["confirmed_shots"].append(shot)
            
            # Register for missed/return detection
            ball_id = shot.get("ball_id")
            if ball_id:
                track = ball_tracker.get_track(ball_id)
                init_dir = utils.compute_ball_direction(track)
                self.missed_detector.register_shot(len(self.shots_log)-1, frame_idx, ball_id, init_dir)

        # 7. Missed/Return Detection
        self.missed_detector.update(frame_idx, ball_tracker, self.shots_log)

        output["total_shots"] = len(self.shots_log)
        output["debug"]["suppressed_balls"] = list(suppressed_balls)
        output["debug"]["active_candidates"] = list(self.shot_tracker.candidates.keys())
        return output
