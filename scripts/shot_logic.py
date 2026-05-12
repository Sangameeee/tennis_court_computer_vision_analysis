"""
Helper functions for managing and recording shot events.
"""
from scripts import utils

def record_shot_event(
    player_id, frame_idx, fps, shot_info, ball_tracker, 
    shots, last_shot_frame, last_shot_index, last_shot_second,
    missed_detector, court_axis
):
    """Record a confirmed shot in JSONL-ready format."""
    shot_entry = {
        "player_id": int(player_id),
        "frame": int(frame_idx),
        "second": round(frame_idx / float(fps), 3) if fps else 0.0,
        "shot": shot_info.get("shot"),
    }
    shots.append(shot_entry)
    shot_idx = len(shots) - 1
    last_shot_frame[player_id] = frame_idx
    last_shot_index[player_id] = shot_idx
    last_shot_second[player_id] = int(frame_idx / fps) if fps else 0
