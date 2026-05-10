"""
Helper functions for visualization and drawing on video frames.
"""
import cv2
from scripts import utils, ball_detection

def draw_court_boundary(frame, poly):
    """Draw court boundary overlay on frame."""
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], (0, 255, 0))
    frame = cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)
    cv2.polylines(frame, [poly], isClosed=True, color=(0, 255, 0), thickness=3)
    return frame

def draw_rackets(frame, rackets):
    """Draw racket detections."""
    for cx, cy, x1, y1, x2, y2, conf, label, cls_id in rackets:
        color = utils.get_class_color(cls_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        tag = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, tag, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_player_info(frame, player_data, shots, last_shot_frame, last_shot_index, fps, frame_idx):
    """Draw person bounding boxes, IDs, and shot info."""
    for data in player_data:
        x1, y1, x2, y2 = data["bbox"]
        player_id = data["player_id"]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, "Person", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        id_text = f"ID:{player_id if player_id is not None else '-'}"
        (tw_id, th_id), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(frame, id_text, (max(x1, x2 - tw_id), max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw shot info
        shot_idx = last_shot_index.get(player_id)
        if shot_idx is not None:
            shot_e = shots[shot_idx]
            last_f = last_shot_frame.get(player_id, -9999)
            missed_f = shot_e.get("missed_frame")
            
            show = (frame_idx - last_f < fps) or (shot_e["status"] == "missed" and missed_f is not None and frame_idx - missed_f < fps)
            if show:
                txt = f"{shot_e['shot']} {shot_e['direction']}" if shot_e['direction'] != "unknown" else shot_e['shot']
                if shot_e["status"] == "missed": txt += " missed"
                cv2.putText(frame, txt, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def draw_ball_visuals(frame, ball_tracker, pixels_per_meter, fps):
    """Draw ball trails and speed/direction overlay."""
    ball_tracker.draw_trails(frame)
    for t in ball_tracker.tracks:
        if t["is_active"] and t["missed"] == 0:
            bx1, by1, bx2, by2 = map(int, t["history"][-1][2:6])
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 165, 255), 2)
            
    pb = ball_detection.get_primary_active_ball(ball_tracker)
    if pb:
        speed_mps = utils.estimate_ball_speed(pb, fps, pixels_per_meter)
        v_dir = utils.get_vertical_direction_label(pb)
        if speed_mps: cv2.putText(frame, f"Speed: {speed_mps:.2f} m/s", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        if v_dir: cv2.putText(frame, f"Dir: {v_dir}", (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
