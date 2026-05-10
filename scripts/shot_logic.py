"""
Helper functions for managing and recording shot events.
"""
from scripts import utils

def record_shot_event(
    player_id, frame_idx, fps, shot_info, ball_tracker, 
    shots, last_shot_frame, last_shot_index, last_shot_second,
    missed_detector, court_axis
):
    """Determine if a shot should be recorded and update shot state."""
    ball_id = shot_info["ball_id"]
    cooldown, current_second = int(fps * 0.6), int(frame_idx / fps)
    last = last_shot_frame.get(player_id, -9999)
    same_second = last_shot_second.get(player_id) == current_second
    
    if not (same_second and frame_idx - last <= cooldown):
        dir_label, dir_vec, ball_speed = "unknown", None, None
        ball_track = ball_tracker.get_track(ball_id) if ball_id is not None else None
        ball_present = ball_track is not None and not ball_track.get("is_dead", False)
        if ball_track and ball_present:
            dir_label, dir_vec = utils.classify_shot_direction(ball_track, court_axis)
            ball_speed = utils.estimate_ball_speed(ball_track, fps, utils.PIXELS_PER_METER)
        
        shot_entry = {
            "player_id": int(player_id), "frame": frame_idx, "second": round(frame_idx / float(fps), 3),
            "shot": shot_info["shot"], "direction": dir_label, "status": "hit",
            "ball_speed_mps": ball_speed, "ball_present": ball_present, "missed_frame": None
        }
        
        existing_idx = last_shot_index.get(player_id)
        if same_second and existing_idx is not None:
            if utils.should_replace_same_second(shots[existing_idx], shot_entry):
                shots[existing_idx] = shot_entry
                last_shot_frame[player_id], last_shot_second[player_id] = frame_idx, current_second
                missed_detector.update_shot(existing_idx, frame_idx, ball_id, dir_vec)
        else:
            shots.append(shot_entry)
            shot_idx = len(shots) - 1
            last_shot_frame[player_id], last_shot_index[player_id], last_shot_second[player_id] = frame_idx, shot_idx, current_second
            missed_detector.register_shot(shot_idx, frame_idx, ball_id, dir_vec)
