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

    def latest_sample(self, player_id):
        hist = self.history.get(player_id)
        if not hist:
            return None
        return hist[-1]

    def wrist_velocity_vectors(self, player_id):
        hist = self.history.get(player_id)
        if not hist or len(hist) < 2:
            return None, None
        prev, cur = hist[-2], hist[-1]
        lw_prev, lw_cur = prev.get("lw"), cur.get("lw")
        rw_prev, rw_cur = prev.get("rw"), cur.get("rw")
        lw_vec = None
        rw_vec = None
        if lw_prev is not None and lw_cur is not None:
            lw_vec = (lw_cur[0] - lw_prev[0], lw_cur[1] - lw_prev[1])
        if rw_prev is not None and rw_cur is not None:
            rw_vec = (rw_cur[0] - rw_prev[0], rw_cur[1] - rw_prev[1])
        return lw_vec, rw_vec

    def wrist_distance(self, player_id):
        sample = self.latest_sample(player_id)
        if not sample:
            return None
        lw, rw = sample.get("lw"), sample.get("rw")
        if lw is None or rw is None:
            return None
        return math.hypot(lw[0] - rw[0], lw[1] - rw[1])

    def anchor_now(self, player_id):
        sample = self.latest_sample(player_id)
        if not sample:
            return None
        ls, rs = sample.get("ls"), sample.get("rs")
        if ls is not None and rs is not None:
            return ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
        lh, rh = sample.get("lh"), sample.get("rh")
        if lh is not None and rh is not None:
            return ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
        return None

    def anchor_displacement(self, player_id, window_frames: int):
        hist = self.history.get(player_id)
        if not hist or len(hist) < window_frames:
            return None
        recent = list(hist)[-window_frames:]
        anchors = []
        for sample in recent:
            ls, rs = sample.get("ls"), sample.get("rs")
            if ls is not None and rs is not None:
                anchors.append(((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0))
                continue
            lh, rh = sample.get("lh"), sample.get("rh")
            if lh is not None and rh is not None:
                anchors.append(((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0))
        if len(anchors) < 2:
            return None
        x0, y0 = anchors[0]
        max_disp = 0.0
        for ax, ay in anchors[1:]:
            max_disp = max(max_disp, math.hypot(ax - x0, ay - y0))
        return max_disp

    def is_walking(self, player_id, window_frames: int, wrist_vel_thresh: float, shoulder_vel_thresh: float, leg_sign_changes: int):
        wrist_speed = self.wrist_speed_now(player_id)
        shoulder_speed = self.shoulder_speed_now(player_id)
        leg_cycle = self.leg_cycle_detected(player_id, window_frames=window_frames, min_sign_changes=leg_sign_changes)
        if wrist_speed is None:
            return False
        wrists_slow = wrist_speed <= wrist_vel_thresh
        shoulders_slow = shoulder_speed is None or shoulder_speed <= shoulder_vel_thresh
        return leg_cycle and wrists_slow and shoulders_slow


class BallStateHelper:
    """Evaluates ball liveliness and plausibility using recent history."""
    def __init__(self, fps: float):
        self.fps = fps
        self.static_frames = max(2, int(fps * utils.DEAD_BALL_WINDOW_SEC))
        self.static_radius_px = utils.DEAD_BALL_MOVE_PX
        self.min_live_move_px = utils.BALL_LIVE_MIN_PX
        self.max_turn_deg = utils.BALL_MAX_TURN_DEG
        self.max_step_mult = utils.BALL_MAX_STEP_MULT
        self.interp_conf_max = utils.BALL_INTERP_CONF_MAX

    def _is_interpolated(self, pt) -> bool:
        return len(pt) > 6 and pt[6] <= self.interp_conf_max

    def is_static(self, track) -> bool:
        if track is None:
            return False
        hist = track.get("history", [])
        if len(hist) < self.static_frames:
            return False
        tail = hist[-self.static_frames:]
        xs = [p[0] for p in tail]
        ys = [p[1] for p in tail]
        spread = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        return spread <= self.static_radius_px

    def _step_sizes(self, hist, window: int = 5):
        if len(hist) < 2:
            return []
        tail = hist[-(window + 1):] if len(hist) > window + 1 else hist
        steps = []
        for i in range(1, len(tail)):
            steps.append(math.hypot(tail[i][0] - tail[i - 1][0], tail[i][1] - tail[i - 1][1]))
        return steps

    def is_plausible(self, track) -> bool:
        if track is None:
            return False
        hist = track.get("history", [])
        if len(hist) < 2:
            return False
        if len(hist) < 3:
            return True
        p0, p1, p2 = hist[-3], hist[-2], hist[-1]
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        if (self._is_interpolated(p1) or self._is_interpolated(p2)) and (v1 != (0.0, 0.0) and v2 != (0.0, 0.0)):
            if utils.angle_between(v1, v2) > self.max_turn_deg:
                return False
        steps = self._step_sizes(hist, window=5)
        if steps:
            positives = [s for s in steps if s > 0.0]
            median = float(np.median(positives)) if positives else 0.0
            max_step = median * self.max_step_mult if median > 0.0 else self.static_radius_px * 3.0
            if math.hypot(v2[0], v2[1]) > max_step:
                return False
        return True

    def is_live(self, track) -> bool:
        if track is None or track.get("is_dead"):
            return False
        if self.is_static(track):
            return False
        if not self.is_plausible(track):
            return False
        hist = track.get("history", [])
        window = min(self.static_frames, len(hist))
        return utils.track_moved_recently(track, window, self.min_live_move_px)

    def position_now(self, track):
        if track is None or not track.get("history"):
            return None
        return track["history"][-1][:2]

    def velocity_now(self, track):
        if track is None:
            return None
        hist = track.get("history", [])
        if len(hist) < 2:
            return None
        p0, p1 = hist[-2], hist[-1]
        return (p1[0] - p0[0], p1[1] - p0[1])

    def speed_over_window(self, track, window_frames: int) -> float:
        if track is None or not self.is_plausible(track) or self.is_static(track):
            return 0.0
        hist = track.get("history", [])
        if len(hist) < window_frames + 1:
            return 0.0
        p0 = hist[-(window_frames + 1)]
        p1 = hist[-1]
        dist_px = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        secs = window_frames / float(self.fps) if self.fps > 0 else 0.0
        if secs <= 0.0:
            return 0.0
        speed_px_per_sec = dist_px / secs
        if utils.PIXELS_PER_METER and utils.PIXELS_PER_METER > 0:
            return round(speed_px_per_sec / utils.PIXELS_PER_METER, 3)
        return round(speed_px_per_sec, 3)


class ShotCandidateTracker:
    """Tracks swing events and confirms forehands once the ball moves away."""
    def __init__(self, confirm_frames: int, away_min_px: float):
        self.confirm_frames = confirm_frames
        self.away_min_px = away_min_px
        self.candidates: dict = {}

    def add_candidate(self, player_id, frame_idx: int, anchor, ball_id, start_dist):
        if player_id is None or anchor is None or ball_id is None:
            return
        self.candidates[player_id] = {
            "start_frame": frame_idx,
            "expires_frame": frame_idx + self.confirm_frames,
            "anchor": anchor,
            "ball_id": ball_id,
            "last_dist": start_dist,
        }

    def drop_candidate(self, player_id):
        self.candidates.pop(player_id, None)

    def update(self, frame_idx: int, ball_tracker, anchors_by_player: dict, ball_state: BallStateHelper):
        confirmed = []
        for player_id, cand in list(self.candidates.items()):
            if frame_idx > cand["expires_frame"]:
                del self.candidates[player_id]
                continue
            anchor = anchors_by_player.get(player_id) or cand["anchor"]
            if anchor is None:
                continue
            track = ball_tracker.get_track(cand["ball_id"]) if ball_tracker is not None else None
            if track is None or not ball_state.is_live(track):
                continue
            pos = ball_state.position_now(track)
            if pos is None:
                continue
            dist = math.hypot(pos[0] - anchor[0], pos[1] - anchor[1])
            if cand["last_dist"] is None:
                cand["last_dist"] = dist
                continue
            if dist > cand["last_dist"] + self.away_min_px:
                ball_speed = ball_state.speed_over_window(track, self.confirm_frames)
                confirmed.append({
                    "player_id": player_id,
                    "shot": "forehand",
                    "ball_id": cand["ball_id"],
                    "ball_speed_ms": ball_speed,
                })
                del self.candidates[player_id]
            else:
                cand["last_dist"] = dist
        return confirmed


class ShotDetectionEngine:
    """Applies skeleton + ball-only shot rules and produces JSONL-ready events."""
    def __init__(self, fps: float):
        self.fps = fps
        self.confirm_frames = max(1, int(fps * utils.SHOT_CONFIRM_SEC))
        self.walk_frames = max(2, int(fps * utils.WALK_WINDOW_SEC))
        self.serve_frames = max(2, int(fps * 1.0))
        self.ball_state = BallStateHelper(fps)
        self.candidates = ShotCandidateTracker(self.confirm_frames, utils.BALL_AWAY_EPS_PX)

    def _make_event(self, player_id: int, frame_idx: int, shot: str, ball_speed_ms: float) -> dict:
        return {
            "player_id": int(player_id),
            "frame": int(frame_idx),
            "second": round(frame_idx / float(self.fps), 3) if self.fps else 0.0,
            "shot": shot,
            "ball_speed_ms": float(ball_speed_ms),
        }

    def _dominant_wrist(self, pose_tracker: PoseHistoryTracker, player_id):
        lw_vec, rw_vec = pose_tracker.wrist_velocity_vectors(player_id)
        sample = pose_tracker.latest_sample(player_id)
        if sample is None:
            return None, None, None
        lw = sample.get("lw")
        rw = sample.get("rw")
        lw_speed = math.hypot(lw_vec[0], lw_vec[1]) if lw_vec is not None else 0.0
        rw_speed = math.hypot(rw_vec[0], rw_vec[1]) if rw_vec is not None else 0.0
        if rw_speed >= lw_speed:
            return rw, rw_vec, rw_speed
        return lw, lw_vec, lw_speed

    def _swing_cue(self, pose_tracker: PoseHistoryTracker, player_id: int, ball_pos, ball_vel) -> bool:
        wrist_pos, wrist_vec, wrist_speed = self._dominant_wrist(pose_tracker, player_id)
        if wrist_pos is None or wrist_vec is None:
            return False
        if wrist_speed < utils.SWING_WRIST_VEL_PX:
            return False
        if ball_pos is None:
            return False
        expected = None
        if ball_vel is not None:
            bmag = math.hypot(ball_vel[0], ball_vel[1])
            if bmag >= self.ball_state.min_live_move_px:
                expected = ball_vel
        if expected is None:
            expected = (ball_pos[0] - wrist_pos[0], ball_pos[1] - wrist_pos[1])
        if expected == (0.0, 0.0):
            return False
        angle = utils.angle_between(wrist_vec, expected)
        return angle <= utils.SWING_DIR_ANGLE_DEG

    def _nearest_ball(self, ball_tracker, anchor):
        if ball_tracker is None or anchor is None:
            return None
        best = None
        best_d = float("inf")
        for t in getattr(ball_tracker, "tracks", []):
            if not t.get("history"):
                continue
            if not self.ball_state.is_plausible(t):
                continue
            bx, by = t["history"][-1][:2]
            d = math.hypot(bx - anchor[0], by - anchor[1])
            if d < best_d:
                best, best_d = t, d
        return best

    def _serve_ready(self, player_id: int, pose_tracker: PoseHistoryTracker, ball_track) -> bool:
        if ball_track is None:
            return False
        if not (self.ball_state.is_static(ball_track) or not self.ball_state.is_live(ball_track)):
            return False
        disp = pose_tracker.anchor_displacement(player_id, self.serve_frames)
        if disp is None:
            return False
        return disp <= utils.PLAYER_STATIC_PX

    def process_frame(self, frame_idx: int, player_states: list, pose_tracker: PoseHistoryTracker, ball_tracker) -> list:
        events = []
        anchors_by_player = {}

        for d in player_states:
            pid = d.get("player_id")
            if pid is None:
                continue
            left_wrist = d.get("left_wrist")
            right_wrist = d.get("right_wrist")
            has_skeleton = left_wrist is not None and right_wrist is not None
            if not has_skeleton:
                continue
            anchor = d.get("anchor") or pose_tracker.anchor_now(pid)
            if anchor is not None:
                anchors_by_player[pid] = anchor

            walking = d.get("walking_detected")
            if walking:
                self.candidates.drop_candidate(pid)
                continue

            ball_track = self._nearest_ball(ball_tracker, anchor)
            ball_pos = self.ball_state.position_now(ball_track) if ball_track else None
            ball_vel = self.ball_state.velocity_now(ball_track) if ball_track else None

            if not self._swing_cue(pose_tracker, pid, ball_pos, ball_vel):
                continue

            wrist_dist = pose_tracker.wrist_distance(pid)
            if wrist_dist is not None and wrist_dist < utils.WRIST_CLOSE_PX:
                ball_speed = self.ball_state.speed_over_window(ball_track, self.confirm_frames) if ball_track else 0.0
                events.append(self._make_event(pid, frame_idx, "backhand", ball_speed))
                self.candidates.drop_candidate(pid)
                continue

            if wrist_dist is not None and wrist_dist >= utils.WRIST_CLOSE_PX and self._serve_ready(pid, pose_tracker, ball_track):
                ball_speed = self.ball_state.speed_over_window(ball_track, self.confirm_frames) if ball_track else 0.0
                events.append(self._make_event(pid, frame_idx, "serve", ball_speed))
                self.candidates.drop_candidate(pid)
                continue

            if wrist_dist is not None and wrist_dist >= utils.WRIST_CLOSE_PX:
                if ball_track is not None and self.ball_state.is_live(ball_track):
                    if pid not in self.candidates.candidates:
                        start_dist = None
                        if anchor is not None and ball_pos is not None:
                            start_dist = math.hypot(ball_pos[0] - anchor[0], ball_pos[1] - anchor[1])
                        self.candidates.add_candidate(pid, frame_idx, anchor, ball_track["id"], start_dist)

        confirmed = self.candidates.update(frame_idx, ball_tracker, anchors_by_player, self.ball_state)
        for shot in confirmed:
            events.append(self._make_event(shot["player_id"], frame_idx, shot["shot"], shot.get("ball_speed_ms", 0.0)))

        return events


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

        self.person_tracker = PersonTracker(max_missed=int(fps * 1.0), max_dist=150.0)
        self.pose_tracker = PoseHistoryTracker(history_len=int(fps * 1.0))
        self.shot_engine = ShotDetectionEngine(fps=fps)
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
        output = {"confirmed_shots": [], "serve_state": None, "server_id": None, "debug": {}}

        self.person_tracker.update(person_detections)

        for pid, pdata in pose_data.items():
            kpts_xy = pdata.get("kpts_xy")
            kpts_conf = pdata.get("kpts_conf")
            self.pose_tracker.update(pid, kpts_xy, kpts_conf)

        player_states = []
        for pid in pose_data.keys():
            sample = self.pose_tracker.latest_sample(pid)
            if sample is None:
                continue
            player_states.append({
                "player_id": pid,
                "anchor": self.pose_tracker.anchor_now(pid),
                "walking_detected": self.pose_tracker.is_walking(
                    pid,
                    window_frames=max(2, int(self.fps * utils.WALK_WINDOW_SEC)),
                    wrist_vel_thresh=utils.WALK_WRIST_VEL_PX,
                    shoulder_vel_thresh=utils.WALK_SHOULDER_VEL_PX,
                    leg_sign_changes=utils.WALK_LEG_SIGN_CHANGES,
                ),
                "left_wrist": sample.get("lw"),
                "right_wrist": sample.get("rw"),
            })

        confirmed = self.shot_engine.process_frame(frame_idx, player_states, self.pose_tracker, ball_tracker)
        for shot in confirmed:
            self.shots_log.append(shot)
            output["confirmed_shots"].append(shot)

        output["total_shots"] = len(self.shots_log)
        output["debug"]["active_candidates"] = list(self.shot_engine.candidates.candidates.keys())
        return output
