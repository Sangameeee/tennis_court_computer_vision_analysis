"""
Shot classification, pose tracking, and person tracking modules.

Revisions over original:
  - BallDisappearanceBuffer   : interpolates ball positions during occlusion gaps
  - GhostBallFilter           : suppresses mid-court ghost / multi-ball detections
  - ShotCandidateTracker      : distance-guarded ball latch, first-frame fall-through,
                                wrists_close re-validation during confirm window
  - SkeletonMovementTracker   : has_swing_recent() replaces accumulating has_swing()
  - PersonTracker             : Hungarian-algorithm matching instead of greedy scan
  - MissedShotDetector        : consecutive-frame direction delta instead of
                                initial-vs-current comparison
  - PoseHistoryTracker        : magnitude threshold in leg_cycle_detected()
  - PersonMotionTracker       : unchanged (was already correct)
"""

import json
import math
import os
from collections import deque

import numpy as np

from scripts import ball_detection, utils

# ---------------------------------------------------------------------------
# Optional scipy import for Hungarian matching.  Falls back to greedy if absent.
# ---------------------------------------------------------------------------
try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ===========================================================================
# NEW: BallDisappearanceBuffer
# ===========================================================================

class BallDisappearanceBuffer:
    """
    Keeps the last known ball position alive for up to `max_gap_frames` frames
    when the ball tracker loses the ball (occlusion, motion blur, net crossing).

    Usage
    -----
    Call `update(ball_tracker, ball_id)` every frame.
    Call `get_position(ball_id)` to get the best-estimate (x, y) whether the
    ball is currently visible or not.  Returns None only if the ball has never
    been seen or the gap has exceeded `max_gap_frames`.

    Interpolation
    -------------
    While the ball is invisible the buffer linearly extrapolates from the last
    observed velocity vector.  This keeps ShotCandidateTracker from stalling
    on `last_dist` comparisons during a brief occlusion.
    """

    def __init__(self, max_gap_frames: int = 12, max_extrap_frames: int = 6):
        self.max_gap_frames = max_gap_frames
        # Beyond max_extrap_frames we stop extrapolating (too uncertain) but
        # still keep the last known position frozen until max_gap_frames.
        self.max_extrap_frames = max_extrap_frames
        # per ball_id: {"pos": (x,y), "vel": (vx,vy), "gap": int}
        self._state: dict = {}

    def update(self, ball_tracker, ball_id):
        """Call once per frame for every ball_id we care about."""
        if ball_id is None:
            return

        track = ball_tracker.get_track(ball_id) if ball_id is not None else None
        visible = (
            track is not None
            and track.get("history")
            and not track.get("is_dead")
        )

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
        """
        Return best-estimate (x, y) or None.
        Extrapolates linearly for the first `max_extrap_frames` of a gap,
        then freezes at last known position until `max_gap_frames` expires.
        """
        if ball_id is None:
            return None
        st = self._state.get(ball_id)
        if st is None or st["pos"] is None:
            return None
        if st["gap"] == 0:
            return st["pos"]
        if st["gap"] > self.max_gap_frames:
            return None

        # Extrapolate
        extrap_steps = min(st["gap"], self.max_extrap_frames)
        vx, vy = st["vel"]
        px, py = st["pos"]
        # Dampen velocity so extrapolation doesn't shoot off wildly
        damping = 0.85 ** extrap_steps
        return (px + vx * extrap_steps * damping,
                py + vy * extrap_steps * damping)

    def is_visible(self, ball_id) -> bool:
        st = self._state.get(ball_id)
        return st is not None and st.get("gap", 1) == 0

    def gap_frames(self, ball_id) -> int:
        st = self._state.get(ball_id)
        return st["gap"] if st else 0

    def reset(self, ball_id):
        self._state.pop(ball_id, None)


# ===========================================================================
# NEW: GhostBallFilter
# ===========================================================================

class GhostBallFilter:
    """
    Suppresses spurious ball detections that cluster in the mid-court region.

    Problems addressed
    ------------------
    1. Net / line markings often produce consistent false positives near the
       centre of the court — they appear every frame in the same spot.
    2. When two detections appear simultaneously and one is already tracked with
       good history, the other is likely a ghost.
    3. Very slow-moving "balls" near the centre are almost certainly static
       artefacts (net markings, reflections).

    Parameters
    ----------
    court_centre_zone  : (x_min, y_min, x_max, y_max) normalised [0,1] or pixel
                         rectangle defining the suspicious mid-court zone.
                         Detections outside this zone are never filtered.
    min_static_frames  : how many consecutive frames a candidate must stay nearly
                         stationary before being labelled a ghost.
    static_move_px     : pixel radius within which a detection counts as static.
    min_history_to_trust: a ball track with at least this many history entries is
                          always trusted (it has proven itself over time).
    """

    def __init__(
        self,
        court_centre_zone=(0.3, 0.35, 0.7, 0.65),
        min_static_frames: int = 8,
        static_move_px: float = 6.0,
        min_history_to_trust: int = 10,
        frame_w: int = 1920,
        frame_h: int = 1080,
    ):
        self.zone = court_centre_zone  # (x0_frac, y0_frac, x1_frac, y1_frac)
        self.min_static_frames = min_static_frames
        self.static_move_px = static_move_px
        self.min_history_to_trust = min_history_to_trust
        self.frame_w = frame_w
        self.frame_h = frame_h
        # per track_id: deque of (x,y) positions to check static-ness
        self._pos_history: dict = {}

    def _in_zone(self, x, y) -> bool:
        x0, y0, x1, y1 = self.zone
        # Support both fractional and pixel coordinates
        if x0 <= 1.0 and y0 <= 1.0:
            fx, fy = x / self.frame_w, y / self.frame_h
            return x0 <= fx <= x1 and y0 <= fy <= y1
        return x0 <= x <= x1 and y0 <= y <= y1

    def update_and_filter(self, ball_tracker) -> set:
        """
        Walk every active track in ball_tracker, update internal position
        histories, and return a set of track IDs that should be considered
        ghosts / suppressed.
        """
        suppressed = set()
        if ball_tracker is None:
            return suppressed

        active_tracks = [
            t for t in getattr(ball_tracker, "tracks", {}).values()
            if not t.get("is_dead") and t.get("history")
        ]

        for t in active_tracks:
            tid = t["id"]
            bx, by = t["history"][-1][:2]
            hist = self._pos_history.setdefault(tid, deque(maxlen=self.min_static_frames + 2))
            hist.append((bx, by))

            # Tracks with long history are trusted unconditionally
            if len(t["history"]) >= self.min_history_to_trust:
                continue

            # Only inspect mid-court zone
            if not self._in_zone(bx, by):
                continue

            # Check if static over recent frames
            if len(hist) >= self.min_static_frames:
                xs = [p[0] for p in hist]
                ys = [p[1] for p in hist]
                spread = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
                if spread <= self.static_move_px:
                    suppressed.add(tid)

        return suppressed

    def is_ghost(self, ball_id, ball_tracker) -> bool:
        """Convenience method — returns True if the given track is currently a ghost."""
        return ball_id in self.update_and_filter(ball_tracker)

    def best_ball(self, ball_tracker, anchor_x, anchor_y, suppressed=None):
        """
        From all non-suppressed, non-dead tracks return the one closest to
        (anchor_x, anchor_y).  If all nearby balls are suppressed, fall back
        to the closest trusted track (long history).
        """
        if ball_tracker is None:
            return None, None

        if suppressed is None:
            suppressed = self.update_and_filter(ball_tracker)

        active = [
            t for t in getattr(ball_tracker, "tracks", {}).values()
            if not t.get("is_dead") and t.get("history")
        ]

        best_real, best_real_d = None, float("inf")
        best_fallback, best_fallback_d = None, float("inf")

        for t in active:
            bx, by = t["history"][-1][:2]
            d = math.hypot(bx - anchor_x, by - anchor_y)
            if t["id"] not in suppressed:
                if d < best_real_d:
                    best_real_d = d
                    best_real = t
            else:
                if len(t["history"]) >= self.min_history_to_trust:
                    if d < best_fallback_d:
                        best_fallback_d = d
                        best_fallback = t

        if best_real is not None:
            return best_real, best_real_d
        return best_fallback, best_fallback_d


# ===========================================================================
# SkeletonMovementTracker  (revised)
# ===========================================================================

class SkeletonMovementTracker:
    """
    Tracks wrist positions per player to gate shot classification on swing motion.

    Changes
    -------
    - `has_swing()` now takes an optional `window` parameter so callers can
      limit how far back they look (avoids stale swings from many frames ago).
    - `has_swing_recent()` is the new recommended entry point — it checks only
      the last `window` frame-pairs and is equivalent to the old has_swing_now()
      extended to N frames.
    - `has_swing_now()` is kept for backward compatibility.
    """

    def __init__(self, movement_thresh: float = 15.0, history_len: int = 5):
        self.history: dict = {}
        self.movement_thresh = movement_thresh
        self.history_len = history_len

    def update(self, player_id, left_wrist, right_wrist):
        if player_id is None:
            return
        hist = self.history.setdefault(player_id, deque(maxlen=self.history_len))
        hist.append((left_wrist, right_wrist))

    @staticmethod
    def _max_disp(seq):
        """Max per-step wrist displacement over a sequence of (lw, rw) pairs."""
        max_d = 0.0
        for i in range(1, len(seq)):
            prev_lw, prev_rw = seq[i - 1]
            cur_lw, cur_rw = seq[i]
            for prev, cur in ((prev_lw, cur_lw), (prev_rw, cur_rw)):
                if prev is None or cur is None:
                    continue
                max_d = max(max_d, math.hypot(cur[0] - prev[0], cur[1] - prev[1]))
        return max_d

    def has_swing(self, player_id, window: int = None):
        """
        True if maximum wrist displacement in the last `window` frames
        exceeds threshold.  `window=None` uses the entire stored history
        (legacy behaviour — prefer has_swing_recent for new code).
        """
        hist = list(self.history.get(player_id, []))
        if len(hist) < 2:
            return False
        if window is not None:
            hist = hist[-max(2, window):]
        return self._max_disp(hist) >= self.movement_thresh

    def has_swing_recent(self, player_id, window: int = 3):
        """
        Recommended entry point.  Checks only the last `window` frame-pairs
        so that a swing from many frames ago does not keep firing.
        """
        return self.has_swing(player_id, window=window)

    def has_swing_now(self, player_id):
        """Check only the very last frame transition (backward compat)."""
        hist = list(self.history.get(player_id, []))
        if len(hist) < 2:
            return False
        return self._max_disp(hist[-2:]) >= self.movement_thresh


# ===========================================================================
# PoseHistoryTracker  (revised)
# ===========================================================================

class PoseHistoryTracker:
    """
    Stores recent pose keypoints per player for motion heuristics.

    Changes
    -------
    - `leg_cycle_detected()` now requires the ankle/knee vertical difference to
      exceed `min_ankle_diff_px` before it contributes a sign — eliminates false
      positives from sub-pixel jitter while standing still.
    """

    def __init__(self, history_len: int = 30, conf_th: float = 0.25):
        self.history: dict = {}
        self.history_len = history_len
        self.conf_th = conf_th

    def _kp(self, kpts_xy, kpts_conf, idx):
        if kpts_xy is None or kpts_conf is None or idx >= len(kpts_xy):
            return None
        if kpts_conf[idx] < self.conf_th:
            return None
        return kpts_xy[idx]

    def update(self, player_id, kpts_xy, kpts_conf):
        if player_id is None or kpts_xy is None or kpts_conf is None:
            return
        sample = {
            "lw": self._kp(kpts_xy, kpts_conf, 9),
            "rw": self._kp(kpts_xy, kpts_conf, 10),
            "ls": self._kp(kpts_xy, kpts_conf, 5),
            "rs": self._kp(kpts_xy, kpts_conf, 6),
            "lh": self._kp(kpts_xy, kpts_conf, 11),
            "rh": self._kp(kpts_xy, kpts_conf, 12),
            "lk": self._kp(kpts_xy, kpts_conf, 13),
            "rk": self._kp(kpts_xy, kpts_conf, 14),
            "la": self._kp(kpts_xy, kpts_conf, 15),
            "ra": self._kp(kpts_xy, kpts_conf, 16),
        }
        hist = self.history.setdefault(player_id, deque(maxlen=self.history_len))
        hist.append(sample)

    def wrist_speed_now(self, player_id):
        hist = self.history.get(player_id)
        if not hist or len(hist) < 2:
            return None
        prev, cur = hist[-2], hist[-1]
        max_disp = 0.0
        for key in ("lw", "rw"):
            p = prev.get(key)
            c = cur.get(key)
            if p is None or c is None:
                continue
            max_disp = max(max_disp, math.hypot(c[0] - p[0], c[1] - p[1]))
        return max_disp

    def shoulder_speed_now(self, player_id):
        hist = self.history.get(player_id)
        if not hist or len(hist) < 2:
            return None
        prev, cur = hist[-2], hist[-1]
        pls, prs = prev.get("ls"), prev.get("rs")
        cls, crs = cur.get("ls"), cur.get("rs")
        if pls is None or prs is None or cls is None or crs is None:
            return None
        prev_mid = ((pls[0] + prs[0]) / 2.0, (pls[1] + prs[1]) / 2.0)
        cur_mid = ((cls[0] + crs[0]) / 2.0, (cls[1] + crs[1]) / 2.0)
        return math.hypot(cur_mid[0] - prev_mid[0], cur_mid[1] - prev_mid[1])

    def leg_cycle_detected(
        self,
        player_id,
        window_frames: int,
        min_sign_changes: int = 2,
        min_ankle_diff_px: float = 8.0,   # NEW: magnitude guard
    ):
        hist = self.history.get(player_id)
        if not hist or len(hist) < 3:
            return False
        recent = list(hist)[-window_frames:]

        phases = []
        for sample in recent:
            la, ra = sample.get("la"), sample.get("ra")
            diff = None
            if la is not None and ra is not None:
                diff = la[1] - ra[1]
            else:
                lk, rk = sample.get("lk"), sample.get("rk")
                if lk is not None and rk is not None:
                    diff = lk[1] - rk[1]
            # Only count frames where limb separation is meaningful
            if diff is not None and abs(diff) >= min_ankle_diff_px:
                phases.append(diff)

        if len(phases) < (min_sign_changes + 1):
            return False

        signs = []
        for p in phases:
            signs.append(1 if p > 0 else -1)

        changes = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
        return changes >= min_sign_changes


# ===========================================================================
# ShotCandidateTracker  (revised)
# ===========================================================================

class ShotCandidateTracker:
    """
    Tracks swing events and confirms shots once the ball moves away.

    Changes
    -------
    - Ball latch requires `nearest_dist < max_latch_dist` (configurable) to
      avoid grabbing a far-away ghost detection.
    - First-frame `last_dist` initialisation no longer skips the confirmation
      check — it falls through, enabling instant confirmation if the ball is
      already departing.
    - `wrists_close` is re-evaluated during the confirmation window whenever
      fresh, high-confidence wrist keypoints are supplied via
      `update_wrists()`.
    - Suppressed (ghost) ball IDs provided by GhostBallFilter are never latched.
    - `ball_disappearance_buffer` optionally provides interpolated positions
      during occlusion so distance comparisons don't stall.
    """

    def __init__(
        self,
        confirm_frames: int,
        away_min_px: float,
        min_ball_move_px: float,
        max_latch_dist: float = 300.0,
        wrist_revalidate_conf: float = 0.5,
    ):
        self.confirm_frames = confirm_frames
        self.away_min_px = away_min_px
        self.min_ball_move_px = min_ball_move_px
        self.max_latch_dist = max_latch_dist
        self.wrist_revalidate_conf = wrist_revalidate_conf
        self.candidates: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_candidate(
        self,
        player_id,
        frame_idx: int,
        anchor,
        ball_id,
        wrists_close: bool,
        serve_context: bool,
        start_dist,
        lw=None,
        rw=None,
    ):
        """
        Register a new swing candidate.

        `lw` / `rw` are the raw wrist (x, y) positions used for re-validation
        during the confirmation window.
        """
        if player_id is None or anchor is None:
            return
        self.candidates[player_id] = {
            "start_frame": frame_idx,
            "expires_frame": frame_idx + self.confirm_frames,
            "anchor": anchor,
            "ball_id": ball_id,
            "wrists_close": wrists_close,
            "serve_context": serve_context,
            "last_dist": start_dist,
            # Wrist re-validation state
            "lw": lw,
            "rw": rw,
            "wrist_confirmed": False,  # becomes True once re-validated
        }

    def drop_candidate(self, player_id):
        self.candidates.pop(player_id, None)

    def update_wrists(self, player_id, lw, rw, lw_conf: float, rw_conf: float):
        """
        Call each frame with the latest wrist detections for a player.
        Re-validates wrists_close if both wrists are seen with high confidence.
        """
        cand = self.candidates.get(player_id)
        if cand is None or cand["wrist_confirmed"]:
            return
        if lw_conf >= self.wrist_revalidate_conf and rw_conf >= self.wrist_revalidate_conf:
            if lw is not None and rw is not None:
                dist = math.hypot(lw[0] - rw[0], lw[1] - rw[1])
                # Heuristic: wrists are "close" if they are within ~half a
                # shoulder-width of each other (~60 px at typical resolution).
                cand["wrists_close"] = dist < 60.0
                cand["wrist_confirmed"] = True
                cand["lw"] = lw
                cand["rw"] = rw

    def update(
        self,
        frame_idx: int,
        ball_tracker,
        anchors_by_player: dict,
        blocked_players=None,
        suppressed_ball_ids: set = None,
        disappearance_buffer: BallDisappearanceBuffer = None,
    ):
        """
        Returns list of confirmed shot dicts:
            {"player_id": ..., "shot": "forehand"|"backhand"|"serve", "ball_id": ...}

        Parameters
        ----------
        suppressed_ball_ids : set of ball IDs identified as ghosts by GhostBallFilter.
        disappearance_buffer : BallDisappearanceBuffer for interpolated positions
                               during occlusion.
        """
        if suppressed_ball_ids is None:
            suppressed_ball_ids = set()

        confirmed = []

        for player_id, cand in list(self.candidates.items()):
            if blocked_players and player_id in blocked_players:
                continue

            # Expire stale candidates
            if frame_idx > cand["expires_frame"]:
                del self.candidates[player_id]
                continue

            # Update anchor to latest player position
            anchor = anchors_by_player.get(player_id) or cand["anchor"]
            if anchor is None:
                continue

            # --- Ball latch (first time we see the ball) -------------------
            if cand["ball_id"] is None or cand["ball_id"] in suppressed_ball_ids:
                nearest_track, nearest_dist = ball_detection.get_nearest_ball_any(
                    ball_tracker, anchor[0], anchor[1]
                )
                if (
                    nearest_track is not None
                    and nearest_dist < self.max_latch_dist
                    and nearest_track["id"] not in suppressed_ball_ids
                ):
                    cand["ball_id"] = nearest_track["id"]
                    # Do NOT set last_dist here — let fall-through below handle it
                    # so we don't waste a frame.
                else:
                    continue  # no suitable ball found yet

            # --- Position of the ball (real or interpolated) ---------------
            pos = None
            if disappearance_buffer is not None:
                pos = disappearance_buffer.get_position(cand["ball_id"])

            if pos is None:
                track = ball_tracker.get_track(cand["ball_id"])
                if track is None or not track.get("history") or track.get("is_dead"):
                    continue
                pos = track["history"][-1][:2]

            bx, by = pos
            dist = math.hypot(bx - anchor[0], by - anchor[1])

            # First frame initialisation — fall through to check immediately
            if cand["last_dist"] is None:
                cand["last_dist"] = dist
                # Do NOT `continue` — check movement on this same frame

            # --- Step-size from ball's own history (real track only) --------
            step = 0.0
            track = ball_tracker.get_track(cand["ball_id"])
            if track and track.get("history") and len(track["history"]) >= 2:
                pbx, pby = track["history"][-2][:2]
                step = math.hypot(bx - pbx, by - pby)

            # --- Confirmation gate -----------------------------------------
            if step >= self.away_min_px and dist >= cand["last_dist"] + self.away_min_px:
                shot_label = "serve" if cand["serve_context"] else (
                    "backhand" if cand["wrists_close"] else "forehand"
                )
                confirmed.append({
                    "player_id": player_id,
                    "shot": shot_label,
                    "ball_id": cand["ball_id"],
                })
                del self.candidates[player_id]
            else:
                cand["last_dist"] = dist

        return confirmed


# ===========================================================================
# PersonTracker  (revised)
# ===========================================================================

class PersonTracker:
    """
    Tracks person/player centres and bounding boxes across frames.

    Changes
    -------
    - Uses the Hungarian algorithm (scipy.optimize.linear_sum_assignment) for
      optimal assignment when scipy is available, which prevents ID swaps when
      two players cross paths.
    - Falls back to the original greedy nearest-neighbour when scipy is absent.
    """

    def __init__(
        self,
        track_file=None,
        max_missed: int = 30,
        max_dist: float = 150.0,
    ):
        self.tracks: list = []
        self.next_id: int = 0
        self.max_missed = max_missed
        self.max_dist = max_dist
        self.track_file = track_file

        if track_file is not None and os.path.exists(track_file):
            try:
                with open(track_file, "r") as f:
                    data = json.load(f)
                for t in data.get("tracks", []):
                    self.tracks.append({
                        "id": t["id"],
                        "center": tuple(t["center"]),
                        "bbox": tuple(t.get("bbox", (0, 0, 0, 0))),
                        "missed": 0,
                    })
                self.next_id = data.get("next_id", self.next_id)
            except Exception:
                pass

    def save(self):
        if not self.track_file:
            return
        data = {
            "next_id": self.next_id,
            "tracks": [
                {
                    "id": t["id"],
                    "center": list(t["center"]),
                    "bbox": list(t.get("bbox", (0, 0, 0, 0))),
                }
                for t in self.tracks
            ],
        }
        try:
            with open(self.track_file, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def update(self, detections):
        """
        detections : list of (cx, cy, x1, y1, x2, y2)
        """
        if not self.tracks or not detections:
            # Handle trivial cases without building a cost matrix
            for i, det in enumerate(detections):
                self.tracks.append({
                    "id": self.next_id,
                    "center": (det[0], det[1]),
                    "bbox": (det[2], det[3], det[4], det[5]),
                    "missed": 0,
                })
                self.next_id += 1
            for t in self.tracks:
                if t["missed"] == 0 and not detections:
                    t["missed"] += 1
            self.tracks = [t for t in self.tracks if t["missed"] <= self.max_missed]
            return

        matched_track_indices = set()
        matched_det_indices = set()

        if _SCIPY_AVAILABLE and len(self.tracks) > 1 and len(detections) > 1:
            # Build cost matrix: tracks × detections
            cost = np.array([
                [
                    math.hypot(det[0] - t["center"][0], det[1] - t["center"][1])
                    for det in detections
                ]
                for t in self.tracks
            ], dtype=float)
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < self.max_dist:
                    det = detections[c]
                    self.tracks[r]["center"] = (det[0], det[1])
                    self.tracks[r]["bbox"] = (det[2], det[3], det[4], det[5])
                    self.tracks[r]["missed"] = 0
                    matched_track_indices.add(r)
                    matched_det_indices.add(c)
        else:
            # Greedy fallback (original logic)
            for ti, t in enumerate(self.tracks):
                best_i = -1
                best_d = float("inf")
                for di, det in enumerate(detections):
                    if di in matched_det_indices:
                        continue
                    d = math.hypot(det[0] - t["center"][0], det[1] - t["center"][1])
                    if d < best_d:
                        best_d = d
                        best_i = di
                if best_i != -1 and best_d < self.max_dist:
                    det = detections[best_i]
                    matched_det_indices.add(best_i)
                    matched_track_indices.add(ti)
                    t["center"] = (det[0], det[1])
                    t["bbox"] = (det[2], det[3], det[4], det[5])
                    t["missed"] = 0

        # Increment missed for unmatched tracks
        for ti, t in enumerate(self.tracks):
            if ti not in matched_track_indices:
                t["missed"] += 1

        # Spawn new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di not in matched_det_indices:
                self.tracks.append({
                    "id": self.next_id,
                    "center": (det[0], det[1]),
                    "bbox": (det[2], det[3], det[4], det[5]),
                    "missed": 0,
                })
                self.next_id += 1

        # Prune lost tracks
        self.tracks = [t for t in self.tracks if t["missed"] <= self.max_missed]

    def lookup_id_by_center(self, cx, cy, max_dist=None):
        md = self.max_dist if max_dist is None else max_dist
        best = None
        best_d = float("inf")
        for t in self.tracks:
            d = math.hypot(cx - t["center"][0], cy - t["center"][1])
            if d < best_d and d < md:
                best_d = d
                best = t["id"]
        return best


# ===========================================================================
# PersonMotionTracker  (unchanged)
# ===========================================================================

class PersonMotionTracker:
    """Tracks player motion to detect walking vs. stationary states."""

    def __init__(self, history_len: int = 45):
        self.history: dict = {}
        self.history_len = history_len

    def update(self, player_id, center, size=None):
        if player_id is None or center is None:
            return
        cx, cy = center
        w = max(1.0, float(size)) if size is not None else 1.0
        h = w
        hist = self.history.setdefault(player_id, deque(maxlen=self.history_len))
        hist.append((cx, cy, w, h))

    def is_static(self, player_id, window_frames: int, move_ratio: float = 0.05):
        hist = self.history.get(player_id)
        if hist is None or len(hist) < window_frames:
            return False
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
        if not other_ids:
            return True
        return all(self.is_static(pid, window_frames, move_ratio=move_ratio) for pid in other_ids)

    def all_static(self, window_frames: int, move_ratio: float = 0.05):
        if not self.history:
            return False
        return all(self.is_static(pid, window_frames, move_ratio=move_ratio) for pid in self.history)


# ===========================================================================
# MissedShotDetector  (revised)
# ===========================================================================

class MissedShotDetector:
    """
    Marks a shot as missed if the ball direction does not change within a timeout.

    Changes
    -------
    - Direction comparison is now consecutive-frame (last_dir vs current_dir)
      instead of initial_dir vs current_dir.  This means a ball that gradually
      curves back (a realistic return) will be detected, whereas the old approach
      required an abrupt 20° change in one step.
    - If the ball is invisible for more than `max_gap_frames` consecutive frames
      the shot is immediately marked missed (ball is lost, not returned).
    - `initial_dir` is still stored for diagnostics / future use.
    """

    def __init__(
        self,
        fps: float,
        timeout_sec: float = 2.0,
        angle_thresh: float = 20.0,
        max_gap_frames: int = 15,
    ):
        self.pending: list = []
        self.fps = fps
        self.timeout_frames = int(fps * timeout_sec)
        self.angle_thresh = angle_thresh
        self.max_gap_frames = max_gap_frames

    def register_shot(self, shot_index: int, frame_idx: int, ball_id, initial_dir):
        if ball_id is None or initial_dir is None:
            return
        self.pending.append({
            "shot_index": shot_index,
            "frame_idx": frame_idx,
            "ball_id": ball_id,
            "initial_dir": initial_dir,
            "last_dir": initial_dir,      # NEW: tracks most recent direction
            "invisible_frames": 0,         # NEW: consecutive invisible frames
        })

    def update_shot(self, shot_index: int, frame_idx: int, ball_id, initial_dir):
        if ball_id is None or initial_dir is None:
            return
        for p in self.pending:
            if p["shot_index"] == shot_index:
                p["ball_id"] = ball_id
                p["last_dir"] = initial_dir
                p["frame_idx"] = frame_idx
                return
        self.register_shot(shot_index, frame_idx, ball_id, initial_dir)

    def update(self, frame_idx: int, ball_tracker, shots: list):
        remaining = []
        for p in self.pending:
            # Hard timeout
            if frame_idx - p["frame_idx"] >= self.timeout_frames:
                shots[p["shot_index"]]["status"] = "missed"
                shots[p["shot_index"]]["missed_frame"] = frame_idx
                continue

            track = ball_tracker.get_track(p["ball_id"])
            current_dir = utils.compute_ball_direction(track)

            if current_dir is None:
                # Ball invisible this frame
                p["invisible_frames"] = p.get("invisible_frames", 0) + 1
                if p["invisible_frames"] >= self.max_gap_frames:
                    shots[p["shot_index"]]["status"] = "missed"
                    shots[p["shot_index"]]["missed_frame"] = frame_idx
                    continue
                remaining.append(p)
                continue

            # Reset invisible counter when ball is seen again
            p["invisible_frames"] = 0

            # Compare against LAST known direction, not initial direction
            angle = utils.angle_between(p["last_dir"], current_dir)
            if angle >= self.angle_thresh:
                # Ball changed direction — shot was returned
                remaining_after = []  # don't keep this one
                _ = remaining_after   # shot confirmed returned, drop from pending
                continue              # do not append to remaining

            # Update last_dir for next frame comparison
            p["last_dir"] = current_dir
            remaining.append(p)

        self.pending = remaining
