"""
Helper functions for extracting detections from YOLO results.
"""
from scripts import utils

def extract_person_detections(base_res, poly):
    """Extract person and racket detections from base YOLO results."""
    person_boxes = []
    person_detections = []
    current_rackets = []
    
    if base_res is not None and base_res.boxes is not None:
        for box in base_res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            label = base_res.names.get(cls_id, str(cls_id)) if base_res.names else str(cls_id)
            
            if not utils.all_corners_inside((x1, y1, x2, y2), poly):
                continue
            
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if cls_id == 0 or label.lower() == "person":
                person_boxes.append((x1, y1, x2, y2))
                person_detections.append((cx, cy, x1, y1, x2, y2))
            else:
                current_rackets.append((cx, cy, x1, y1, x2, y2, conf, label, cls_id))
                
    return person_boxes, person_detections, current_rackets

def extract_ball_detections(ball_res, poly):
    """Extract ball detections from ball-specific YOLO results."""
    current_balls = []
    if ball_res is not None and ball_res.boxes is not None:
        for box in ball_res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0]) if box.cls is not None else 0
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            label = ball_res.names.get(cls_id, "ball") if ball_res.names else "ball"
            if cls_id != 0 and "ball" not in label.lower():
                continue
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if not utils.center_inside((cx, cy), poly):
                continue
            area = max(1.0, (x2 - x1) * (y2 - y1))
            current_balls.append((cx, cy, x1, y1, x2, y2, conf, area))
    return current_balls
