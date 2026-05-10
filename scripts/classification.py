"""
Shot classification, pose tracking, and person tracking modules.
"""
import json
import os
import math
from collections import deque
from scripts import utils, ball_detection


class SkeletonMovementTracker:
    """Tracks wrist positions per player to gate shot classification on swing motion."""

    def __init__(self, movement_thresh=15.0, history_len=5):
        self.history = {}
        self.movement_thresh = movement_thresh
        self.history_len = history_len

    def update(self, player_id, left_wrist, right_wrist):
        if player_id is None:
            return
        hist = self.history.setdefault(player_id, deque(maxlen=self.history_len))
        hist.append((left_wrist, right_wrist))

    def has_swing(self, player_id):
        hist = self.history.get(player_id)
        if not hist or len(hist) < 2:
            return False
        max_disp = 0.0
        for i in range(1, len(hist)):
            prev_lw, prev_rw = hist[i - 1]
            cur_lw, cur_rw = hist[i]
            for prev, cur in ((prev_lw, cur_lw), (prev_rw, cur_rw)):
                if prev is None or cur is None:
                    continue
                disp = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
                max_disp = max(max_disp, disp)
        return max_disp >= self.movement_thresh

    def has_swing_now(self, player_id):
        hist = self.history.get(player_id)
        if not hist or len(hist) < 2:
            return False
        prev_lw, prev_rw = hist[-2]
        cur_lw, cur_rw = hist[-1]
        max_disp = 0.0
        for prev, cur in ((prev_lw, cur_lw), (prev_rw, cur_rw)):
            if prev is None or cur is None:
                continue
            disp = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
            max_disp = max(max_disp, disp)
        return max_disp >= self.movement_thresh


class PoseHistoryTracker:
    """Stores recent pose keypoints per player for motion heuristics."""

    def __init__(self, history_len=30, conf_th=0.25):
        self.history = {}
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

    def leg_cycle_detected(self, player_id, window_frames, min_sign_changes=2):
        hist = self.history.get(player_id)
        if not hist or len(hist) < 3:
            return False
        recent = list(hist)[-window_frames:]

        phases = []
        for sample in recent:
            la, ra = sample.get("la"), sample.get("ra")
            if la is not None and ra is not None:
                phases.append(la[1] - ra[1])
                continue
            lk, rk = sample.get("lk"), sample.get("rk")
            if lk is not None and rk is not None:
                phases.append(lk[1] - rk[1])

        if len(phases) < (min_sign_changes + 1):
            return False

        signs = []
        for p in phases:
            if abs(p) < 1e-3:
                continue
            signs.append(1 if p > 0 else -1)
        if len(signs) < (min_sign_changes + 1):
            return False

        changes = 0
        for i in range(1, len(signs)):
            if signs[i] != signs[i - 1]:
                changes += 1
        return changes >= min_sign_changes


class ShotCandidateTracker:
    """Tracks swing events and confirms shots once the ball moves away."""

    def __init__(self, confirm_frames, away_min_px, min_ball_move_px):
        self.confirm_frames = confirm_frames
        self.away_min_px = away_min_px
        self.min_ball_move_px = min_ball_move_px
        self.candidates = {}

    def add_candidate(self, player_id, frame_idx, anchor, ball_id, wrists_close, serve_context, start_dist):
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
        }

    def drop_candidate(self, player_id):
        if player_id in self.candidates:
            del self.candidates[player_id]

    def update(self, frame_idx, ball_tracker, anchors_by_player, blocked_players=None):
        confirmed = []
        for player_id, cand in list(self.candidates.items()):
            if blocked_players and player_id in blocked_players:
                continue
            if frame_idx > cand["expires_frame"]:
                del self.candidates[player_id]
                continue
            anchor = anchors_by_player.get(player_id) or cand["anchor"]
            if anchor is None:
                continue
            if cand["ball_id"] is None:
                nearest_track, nearest_dist = ball_detection.get_nearest_ball_any(
                    ball_tracker,
                    anchor[0],
                    anchor[1],
                )
                if nearest_track is not None:
                    cand["ball_id"] = nearest_track["id"]
                    cand["last_dist"] = nearest_dist
            track = ball_tracker.get_track(cand["ball_id"]) if cand["ball_id"] is not None else None
            if track is None or not track.get("history"):
                continue
            if track.get("is_dead"):
                continue
            bx, by = track["history"][-1][:2]
            dist = math.hypot(bx - anchor[0], by - anchor[1])
            if cand["last_dist"] is None:
                cand["last_dist"] = dist
                continue
            step = 0.0
            if len(track["history"]) >= 2:
                pbx, pby = track["history"][-2][:2]
                step = math.hypot(bx - pbx, by - pby)
            if step >= self.away_min_px and dist >= cand["last_dist"] + self.away_min_px:
                shot_label = "serve" if cand["serve_context"] else (
                    "backhand" if cand["wrists_close"] else "forehand"
                )
                confirmed.append(
                    {
                        "player_id": player_id,
                        "shot": shot_label,
                        "ball_id": cand["ball_id"],
                    }
                )
                del self.candidates[player_id]
            else:
                cand["last_dist"] = dist
        return confirmed


class PersonTracker:
    """Tracks person/player centers and bounding boxes across frames."""

    def __init__(self, track_file=None, max_missed=30, max_dist=150):
        self.tracks = []
        self.next_id = 0
        self.max_missed = max_missed
        self.max_dist = max_dist
        self.track_file = track_file
        if track_file is not None and os.path.exists(track_file):
            try:
                with open(track_file, "r") as f:
                    data = json.load(f)
                for t in data.get("tracks", []):
                    self.tracks.append(
                        {
                            "id": t["id"],
                            "center": tuple(t["center"]),
                            "bbox": tuple(t.get("bbox", (0, 0, 0, 0))),
                            "missed": 0,
                        }
                    )
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
        matched = set()
        for t in self.tracks:
            best_i = -1
            best_d = float("inf")
            for i, det in enumerate(detections):
                if i in matched:
                    continue
                d = math.hypot(det[0] - t["center"][0], det[1] - t["center"][1])
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i != -1 and best_d < self.max_dist:
                det = detections[best_i]
                matched.add(best_i)
                t["center"] = (det[0], det[1])
                t["bbox"] = (det[2], det[3], det[4], det[5])
                t["missed"] = 0
            else:
                t["missed"] += 1

        for i, det in enumerate(detections):
            if i in matched:
                continue
            self.tracks.append(
                {
                    "id": self.next_id,
                    "center": (det[0], det[1]),
                    "bbox": (det[2], det[3], det[4], det[5]),
                    "missed": 0,
                }
            )
            self.next_id += 1

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


class PersonMotionTracker:
    """Tracks player motion to detect walking vs. stationary states."""

    def __init__(self, history_len=45):
        self.history = {}
        self.history_len = history_len

    def update(self, player_id, center, size=None):
        if player_id is None or center is None:
            return
        cx, cy = center
        if size is None:
            w = 1.0
            h = 1.0
        else:
            w = max(1.0, float(size))
            h = max(1.0, float(size))
        hist = self.history.setdefault(player_id, deque(maxlen=self.history_len))
        hist.append((cx, cy, w, h))

    def is_static(self, player_id, window_frames, move_ratio=0.05):
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

    def others_static(self, current_id, window_frames, move_ratio=0.05):
        other_ids = [pid for pid in self.history.keys() if pid != current_id]
        if not other_ids:
            return True
        for pid in other_ids:
            if not self.is_static(pid, window_frames, move_ratio=move_ratio):
                return False
        return True

    def all_static(self, window_frames, move_ratio=0.05):
        if not self.history:
            return False
        for pid in self.history.keys():
            if not self.is_static(pid, window_frames, move_ratio=move_ratio):
                return False
        return True


class MissedShotDetector:
    """Marks a shot as missed if ball direction does not change within a timeout."""

    def __init__(self, fps, timeout_sec=2.0, angle_thresh=20.0):
        self.pending = []
        self.fps = fps
        self.timeout_frames = int(fps * timeout_sec)
        self.angle_thresh = angle_thresh

    def register_shot(self, shot_index, frame_idx, ball_id, initial_dir):
        if ball_id is None or initial_dir is None:
            return
        self.pending.append(
            {
                "shot_index": shot_index,
                "frame_idx": frame_idx,
                "ball_id": ball_id,
                "initial_dir": initial_dir,
            }
        )

    def update_shot(self, shot_index, frame_idx, ball_id, initial_dir):
        if ball_id is None or initial_dir is None:
            return
        for p in self.pending:
            if p["shot_index"] == shot_index:
                p["ball_id"] = ball_id
                p["initial_dir"] = initial_dir
                p["frame_idx"] = frame_idx
                return
        self.register_shot(shot_index, frame_idx, ball_id, initial_dir)

    def update(self, frame_idx, ball_tracker, shots):
        remaining = []
        for p in self.pending:
            if frame_idx - p["frame_idx"] >= self.timeout_frames:
                shots[p["shot_index"]]["status"] = "missed"
                shots[p["shot_index"]]["missed_frame"] = frame_idx
                continue

            track = ball_tracker.get_track(p["ball_id"])
            current_dir = utils.compute_ball_direction(track)
            if current_dir is not None:
                angle = utils.angle_between(p["initial_dir"], current_dir)
                if angle >= self.angle_thresh:
                    continue
            remaining.append(p)
        self.pending = remaining
