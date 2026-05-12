"""
Helper functions for analyzing individual players in each frame.
"""
import math
from scripts import utils, ball_detection

def process_player_frame(
    player_id, x1, y1, x2, y2, player_cx, player_cy, 
    h, w, frame, orig, pose_model,
    movement_tracker, pose_history, person_motion_tracker,
    ball_tracker, ball_relevance_window_frames, walk_window_frames
):
    """Analyze a single player's pose, motion, and interaction with the ball."""
    is_near_side = player_cy > (h / 2.0)
    swing_detected = False
    has_skeleton = False
    anchor = None
    shoulder_span = None
    hip_span = None
    wrists_close = False
    left_wrist = None
    right_wrist = None
    left_wrist_conf = None
    right_wrist_conf = None
    wrist_distance_px = None
    p_kpts = None
    p_kpts_cf = None
    
    pad_w, pad_h = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
    cx1, cy1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
    cx2, cy2 = min(w, x2 + pad_w), min(h, y2 + pad_h)
    
    crop = orig[cy1:cy2, cx1:cx2]
    if crop.size > 0:
        pose_res = pose_model.predict(crop, verbose=False)[0]
        if pose_res.keypoints is not None and len(pose_res.keypoints) > 0:
            p_kpts = pose_res.keypoints.xy.cpu().numpy()[0]
            p_kpts_cf = pose_res.keypoints.conf.cpu().numpy()[0]
            for k_idx in range(len(p_kpts)):
                if p_kpts_cf[k_idx] > 0:
                    p_kpts[k_idx][0] += cx1
                    p_kpts[k_idx][1] += cy1
            
            utils.draw_pose(frame, p_kpts, p_kpts_cf, color=(0, 255, 255), conf_th=0.25)
            has_skeleton = True
            left_wrist = utils.get_keypoint(p_kpts, p_kpts_cf, 9, conf_th=0.25)
            right_wrist = utils.get_keypoint(p_kpts, p_kpts_cf, 10, conf_th=0.25)
            left_wrist_conf = float(p_kpts_cf[9]) if p_kpts_cf is not None and len(p_kpts_cf) > 9 else None
            right_wrist_conf = float(p_kpts_cf[10]) if p_kpts_cf is not None and len(p_kpts_cf) > 10 else None
            movement_tracker.update(player_id, left_wrist, right_wrist)
            pose_history.update(player_id, p_kpts, p_kpts_cf)
            swing_detected = movement_tracker.has_swing_now(player_id)
            
            ls, rs = utils.get_keypoint(p_kpts, p_kpts_cf, 5), utils.get_keypoint(p_kpts, p_kpts_cf, 6)
            lh, rh = utils.get_keypoint(p_kpts, p_kpts_cf, 11), utils.get_keypoint(p_kpts, p_kpts_cf, 12)
            if ls is not None and rs is not None: shoulder_span = math.hypot(ls[0]-rs[0], ls[1]-rs[1])
            if lh is not None and rh is not None: hip_span = math.hypot(lh[0]-rh[0], lh[1]-rh[1])
            
            ref_len = shoulder_span if shoulder_span is not None else hip_span
            wrists_close = utils.wrists_close_enough(left_wrist, right_wrist, ref_len=ref_len, min_px=utils.WRIST_CLOSE_PX, ratio=utils.WRIST_CLOSE_RATIO)
            if left_wrist is not None and right_wrist is not None:
                wrist_distance_px = math.hypot(left_wrist[0] - right_wrist[0], left_wrist[1] - right_wrist[1])
            anchor = utils.get_player_anchor(p_kpts, p_kpts_cf)
            motion_span = shoulder_span if shoulder_span is not None else hip_span
            person_motion_tracker.update(player_id, anchor, size=motion_span)
    
    # Check ball relevance
    nearest_ball_track, nearest_ball_dist = None, None
    ball_relevant = False
    if anchor is not None:
        nearest_ball_track, nearest_ball_dist = ball_detection.get_nearest_ball_any(ball_tracker, anchor[0], anchor[1])
        if nearest_ball_track is not None and nearest_ball_track.get("is_dead"):
            nearest_ball_track, nearest_ball_dist = None, None
        if nearest_ball_track is not None:
            span = shoulder_span if shoulder_span is not None else hip_span
            if span is not None:
                radius = max(utils.BALL_NEAR_MIN_PX, span * utils.BALL_NEAR_RADIUS_RATIO)
                near_recent = ball_detection.ball_near_recent(nearest_ball_track, anchor, radius, ball_relevance_window_frames)
                heading_toward = ball_detection.ball_heading_toward(nearest_ball_track, anchor, ball_relevance_window_frames, utils.BALL_TOWARD_ANGLE_DEG, utils.BALL_MOVE_MIN_PX)
                ball_relevant = near_recent or heading_toward
    
    walking_detected = False
    if has_skeleton and player_id is not None:
        walking_detected = pose_history.is_walking(
            player_id,
            window_frames=walk_window_frames,
            wrist_vel_thresh=utils.WALK_WRIST_VEL_PX,
            shoulder_vel_thresh=utils.WALK_SHOULDER_VEL_PX,
            leg_sign_changes=utils.WALK_LEG_SIGN_CHANGES,
        )
            
    return {
        "player_id": player_id, "bbox": (x1, y1, x2, y2), "center": (player_cx, player_cy),
        "is_near_side": is_near_side, "has_skeleton": has_skeleton, "anchor": anchor,
        "swing_detected": swing_detected, "wrists_close": wrists_close, "walking_detected": walking_detected,
        "ball_relevant": ball_relevant, "ball_id": nearest_ball_track["id"] if ball_relevant and nearest_ball_track else None,
        "ball_start_dist": nearest_ball_dist if ball_relevant else None,
        "left_wrist": left_wrist, "right_wrist": right_wrist,
        "left_wrist_conf": left_wrist_conf, "right_wrist_conf": right_wrist_conf,
        "wrist_distance_px": wrist_distance_px
    }
