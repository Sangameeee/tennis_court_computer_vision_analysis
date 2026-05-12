"""
Orchestrates the video processing pipeline.
"""
import cv2
import os
import json
import shutil
from ultralytics import YOLO
from scripts import utils, ball_detection, classification
from scripts import detection_helper, drawing_helper, player_processor

class VideoProcessor:
    """Orchestrates the video processing pipeline."""

    def __init__(self, config):
        """Initialize with model paths and settings."""
        self.config = config
        print("Loading YOLO models...")
        self.model = YOLO(config.get("model_path", "yolo26x.pt"))
        self.model_ball = YOLO(config.get("model_ball_path", "last.pt"))
        self.pose_model = YOLO(config.get("pose_model_path", "yolov8x-pose.pt"))
        self._inject_constants()

    def _inject_constants(self):
        """Inject configuration constants into the utils module."""
        c = self.config
        utils.PIXELS_PER_METER = c.get("PIXELS_PER_METER", 100.0)
        utils.SERVE_NO_BALL_SEC = c.get("SERVE_NO_BALL_SEC", 1.0)
        utils.PLAYER_STATIC_RATIO = c.get("PLAYER_STATIC_RATIO", 0.05)
        utils.BALL_TURN_ANGLE_DEG = c.get("BALL_TURN_ANGLE_DEG", 25.0)
        utils.WRIST_CLOSE_RATIO = c.get("WRIST_CLOSE_RATIO", 0.0)
        utils.WRIST_CLOSE_PX = c.get("WRIST_CLOSE_PX", 4.0)
        utils.BALL_MOVE_MIN_PX = c.get("BALL_MOVE_MIN_PX") or max(8.0, utils.PIXELS_PER_METER * 0.05)
        utils.BALL_LIVE_MIN_PX = c.get("BALL_LIVE_MIN_PX") or max(2.0, utils.BALL_MOVE_MIN_PX * 0.25)
        utils.BALL_TOWARD_ANGLE_DEG = c.get("BALL_TOWARD_ANGLE_DEG", 35.0)
        utils.WALK_WINDOW_SEC = c.get("WALK_WINDOW_SEC", 1.0)
        utils.WALK_WRIST_VEL_PX = c.get("WALK_WRIST_VEL_PX", 4.0)
        utils.WALK_SHOULDER_VEL_PX = c.get("WALK_SHOULDER_VEL_PX", 5.0)
        utils.WALK_LEG_SIGN_CHANGES = c.get("WALK_LEG_SIGN_CHANGES", 2)
        utils.INTERP_MAX_SEC = c.get("INTERP_MAX_SEC", 0.5)
        utils.SHOT_CONFIRM_SEC = c.get("SHOT_CONFIRM_SEC", 0.5)
        utils.DEAD_BALL_WINDOW_SEC = c.get("DEAD_BALL_WINDOW_SEC", 1.0)
        utils.DEAD_BALL_MOVE_PX = c.get("DEAD_BALL_MOVE_PX") or max(6.0, utils.PIXELS_PER_METER * 0.03)
        utils.BALL_MAX_TURN_DEG = c.get("BALL_MAX_TURN_DEG", 120.0)
        utils.BALL_MAX_STEP_MULT = c.get("BALL_MAX_STEP_MULT", 3.0)
        utils.BALL_INTERP_CONF_MAX = c.get("BALL_INTERP_CONF_MAX", 0.01)
        utils.PLAYER_STATIC_PX = c.get("PLAYER_STATIC_PX") or max(12.0, utils.PIXELS_PER_METER * 0.1)
        utils.BALL_AWAY_EPS_PX = c.get("BALL_AWAY_EPS_PX", 1.0)
        utils.SWING_WRIST_VEL_PX = c.get("SWING_WRIST_VEL_PX", 18.0)
        utils.SWING_DIR_ANGLE_DEG = c.get("SWING_DIR_ANGLE_DEG", 60.0)
        utils.BALL_NEAR_RADIUS_RATIO = c.get("BALL_NEAR_RADIUS_RATIO", 1.6)
        utils.BALL_NEAR_MIN_PX = c.get("BALL_NEAR_MIN_PX", 40.0)
        utils.BALL_AWAY_MIN_PX = c.get("BALL_AWAY_MIN_PX", 2.0)
        utils.BALL_RELEVANCE_SEC = c.get("BALL_RELEVANCE_SEC", 1.0)

    def process(self):
        """Run the full video processing pipeline."""
        save_root, temp_root, drive_available = utils.resolve_save_dirs(self.config.get("DRIVE_SAVE_PATH", ""))
        video_input = self.config.get("video_path_input", "")
        output_video_path = os.path.join(save_root, "output_video_with_filtered_detections.mp4")
        write_video_path = os.path.join(temp_root, "output_video_with_filtered_detections.mp4") if drive_available else output_video_path
        
        cap_input = cv2.VideoCapture(video_input)
        if not cap_input.isOpened(): raise FileNotFoundError(f"Video not found: {video_input}")
        
        fps = int(cap_input.get(cv2.CAP_PROP_FPS)) or 30
        frame_size = (int(cap_input.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        writer = None
        for fourcc in ["avc1", "H264", "mp4v"]:
            w = cv2.VideoWriter(write_video_path, cv2.VideoWriter_fourcc(*fourcc), fps, frame_size)
            if w.isOpened(): 
                writer = w
                break
        if not writer: raise IOError(f"Could not open writer for {write_video_path}")
        
        interp_max = max(1, int(fps * utils.INTERP_MAX_SEC))
        ball_tracker = ball_detection.BallTracker(max_interpolate_frames=interp_max, dead_window_frames=max(2, int(fps * utils.DEAD_BALL_WINDOW_SEC)), dead_move_px=utils.DEAD_BALL_MOVE_PX)
        movement_tracker = classification.SkeletonMovementTracker()
        person_motion_tracker = classification.PersonMotionTracker(history_len=int(fps * 1.5))
        pose_history = classification.PoseHistoryTracker(history_len=int(fps * 1.5))
        shot_engine = classification.ShotDetectionEngine(fps=fps)
        person_tracker = classification.PersonTracker(track_file=os.path.join(save_root, ".players.track"))
        
        shots, last_shot_frame, last_shot_index = [], {}, {}
        frame_idx = 0
        ball_rel_frames = max(2, int(fps * utils.BALL_RELEVANCE_SEC))
        walk_frames = max(2, int(fps * utils.WALK_WINDOW_SEC))
        
        while cap_input.isOpened():
            ret, orig = cap_input.read()
            if not ret: break
            frame, h, w = orig.copy(), orig.shape[0], orig.shape[1]
            frame = drawing_helper.draw_court_boundary(frame, utils.POLYGON_POINTS_FOR_DRAW_AND_TEST)
            
            base_res = self.model.track(frame, persist=True, verbose=False, classes=[0, 38])
            ball_res = self.model_ball.track(frame, persist=True, verbose=False)
            
            p_boxes, p_dets, rackets = detection_helper.extract_person_detections(base_res[0] if base_res else None, utils.POLYGON_POINTS_FOR_DRAW_AND_TEST)
            c_balls = detection_helper.extract_ball_detections(ball_res[0] if ball_res else None, utils.POLYGON_POINTS_FOR_DRAW_AND_TEST)
            
            drawing_helper.draw_rackets(frame, rackets)
            ball_tracker.update(c_balls, frame_idx)
            person_tracker.update(p_dets)
            
            p_data = []
            for pb in p_boxes:
                px1, py1, px2, py2 = map(int, pb)
                pid = person_tracker.lookup_id_by_center((px1 + px2) / 2.0, (py1 + py2) / 2.0)
                p_data.append(player_processor.process_player_frame(pid, px1, py1, px2, py2, (px1+px2)/2.0, (py1+py2)/2.0, h, w, frame, orig, self.pose_model, movement_tracker, pose_history, person_motion_tracker, ball_tracker, ball_rel_frames, walk_frames))
            
            shot_events = shot_engine.process_frame(frame_idx, p_data, pose_history, ball_tracker)
            for ev in shot_events:
                shots.append(ev)
                pid = ev.get("player_id")
                if pid is not None:
                    last_shot_frame[pid] = ev.get("frame", frame_idx)
                    last_shot_index[pid] = len(shots) - 1
            
            drawing_helper.draw_player_info(frame, p_data, shots, last_shot_frame, last_shot_index, fps, frame_idx)
            drawing_helper.draw_ball_visuals(frame, ball_tracker, utils.PIXELS_PER_METER, fps)
            writer.write(frame)
            if frame_idx % 100 == 0: print(f"Processed {frame_idx} frames...")
            frame_idx += 1
            
        cap_input.release()
        writer.release()
        if drive_available: shutil.copy2(write_video_path, output_video_path)
        person_tracker.save()
        
        shots_path = os.path.join(save_root, "shots.jsonl")
        with open(shots_path, "w") as f:
            for s in shots:
                f.write(json.dumps({
                    "player_id": utils.safe_int(s.get("player_id")),
                    "frame": utils.safe_int(s.get("frame")),
                    "second": utils.safe_float(s.get("second")),
                    "shot": s.get("shot"),
                    "ball_speed_ms": utils.safe_float(s.get("ball_speed_ms")),
                }) + "\n")
        print(f"Done. Saved to: {output_video_path} and {shots_path}")
