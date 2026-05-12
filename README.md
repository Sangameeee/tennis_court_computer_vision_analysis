# Tennis Shot Detection Pipeline

A computer vision system that detects tennis players, tracks ball movement, estimates player poses, and identifies shot types (forehand, backhand, serve) in video footage.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Tools & Libraries Used](#tools--libraries-used)
3. [System Architecture](#system-architecture)
4. [Main Tasks & How They Work](#main-tasks--how-they-work)
   - [Task 1: Person Detection](#task-1-person-detection)
   - [Task 2: Ball Detection & Tracking](#task-2-ball-detection--tracking)
   - [Task 3: Pose Estimation](#task-3-pose-estimation)
   - [Task 4: Shot Detection](#task-4-shot-detection)
   - [Task 5: Motion & Movement Analysis](#task-5-motion--movement-analysis)
5. [Processing Pipeline](#processing-pipeline)
6. [Output Format](#output-format)
7. [Key Concepts](#key-concepts)
8. [Configuration](#configuration)

---

## Introduction

This project analyzes tennis videos to automatically detect and classify shots. It processes video frames in real-time using:
- **AI models** to detect people and balls
- **Skeleton detection** to understand player pose
- **Motion analysis** to identify swings and movements
- **Ball tracking** to follow ball trajectory

The system outputs a JSON file with all detected shots, including:
- Player ID
- Shot type (forehand, backhand, serve)
- Frame number and timestamp

---

## Tools & Libraries Used

| Tool | Purpose |
|------|---------|
|**YOLO26x**| Object detection for persons |
| **YOLOv8** (Ultralytics) | Object detection  balls |
| **YOLOv8-Pose** | Skeleton/keypoint detection (body joints) |
| **OpenCV (cv2)** | Video reading/writing, image processing, drawing |
| **NumPy** | Numerical computations and array operations |
| **JSON** | Storing shot detection results |

---

## System Architecture

```
Video Input
    ↓
[Frame Processing Loop]
    ├─→ Person Detection (YOLO)
    ├─→ Ball Detection (YOLO)
    ├─→ Pose Estimation (YOLOv8-Pose)
    ├─→ Shot Detection Engine
    └─→ Video Output with Annotations
    ↓
shots.jsonl (Detection Results)
```

---

## Main Tasks & How They Work

### Task 1: Person Detection

**What it does:** Finds and tracks all players (people) in each video frame.

**How it works:**

1. **Detection** - For every frame:
   - YOLOv26 model scans the entire frame
   - Identifies objects and classifies them as "person" (class ID = 0)
   - Returns bounding boxes: `(x1, y1, x2, y2)` - top-left and bottom-right corners

2. **Boundary Check** - Keeps only detections inside the court:
   - Uses court polygon boundary to filter out detections outside the court
   - Ensures we only track players on the court

3. **ID Assignment** - Assigns persistent IDs to each player:
   - The `PersonTracker` matches detections across frames
   - Same person gets the same ID across frames
   - If a person disappears for a few frames, they get a new ID when they reappear

**Key File:** `scripts/detection_helper.py` - `extract_person_detections()`

---

### Task 2: Ball Detection & Tracking

**What it does:** Finds the ball and follows its movement across frames.

**How it works:**

1. **Detection** - Ball Detection YOLO model (specialized model trained on tennis balls):
   - Finds all ball-like objects in each frame
   - Returns confidence score and bounding box
   - Filters out non-ball detections

2. **Tracking** - The `BallTracker` maintains continuous ball trajectory:
   - Matches current frame's ball to previous frame's ball
   - Calculates distance between consecutive positions
   - If distance is reasonable, it's the same ball (same track)
   - If ball is lost for a few frames, it interpolates (predicts position)

3. **Interpolation** - When ball is briefly not detected:
   - Uses velocity from previous frames
   - Predicts where ball should be
   - Marks these as interpolated (confidence = 0)

4. **Dead Ball Detection** - Identifies when ball is stationary:
   - Checks if ball moved less than `DEAD_BALL_MOVE_PX` pixels in recent frames
   - Marks ball as "dead" (not in play)
   - Live balls must move a minimum distance to be considered moving

**Key File:** `scripts/ball_detection.py` - `BallTracker` class

---

### Task 3: Pose Estimation
For this if I sent whole frames only the first two players skeleton was detected and far away players skeleton couldnt be detected
so from the base model I just sent the cut the full frames into boundaries of the players and sent those cut image for detections to skeleton model 

**What it does:** Detects body joints (skeleton) of each player.

**How it works:**

1. **Keypoint Detection** - YOLOv8-Pose model identifies 17 body joints:
   - Head, shoulders, elbows, wrists, hips, knees, ankles
   - Returns `(x, y)` coordinates for each joint
   - Returns confidence score for each joint

2. **Confidence Filtering** - Only uses reliable detections:
   - Joint must have confidence > 0.25 to be used
   - Low-confidence joints are ignored

3. **Key Joint Tracking** - We specifically focus on:
   - **Left Wrist** (keypoint 9)
   - **Right Wrist** (keypoint 10)
   - **Shoulders** - To detect if player is walking
   - **Hips** - To detect body orientation

4. **Drawing** - Skeleton is drawn on video:
   - Lines connect joints together
   - Shows player's body posture

**Key File:** `scripts/player_processor.py` - `process_player_frame()`

---

### Task 4: Shot Detection

**What it does:** Identifies when and what type of shot a player makes.

**How it works:**

The system uses multiple rules to detect shots:

#### A. **Swing Detection (Wrist-based)**

Detects when a player is swinging:

1. **Wrist Velocity Check:**
   - Calculates speed of dominant wrist (left or right, whichever moves faster)
   - If wrist speed > `SWING_WRIST_VEL_PX` pixels/frame → swing happening

2. **Wrist Direction Check:**
   - Compares wrist movement direction with ball direction
   - Angle must be < `SWING_DIR_ANGLE_DEG` (60°)
   - This ensures wrist is moving toward the ball, not away

#### B. **Shot Classification - 3 Types**

**Backhand Shot:**
- Wrists are close together (< `WRIST_CLOSE_PX` pixels)
- Wrist is moving fast
- Ball is nearby

**Serve:**
- Wrists are far apart (> `WRIST_CLOSE_PX` pixels)
- Player is NOT walking
- Player is static (OR has no sharp wrist movement for last 1 second)
- Ball is stationary or slow-moving
- Confirms serve is starting

**Forehand Shot:**
- Wrists are far apart
- Wrist is moving fast
- Ball changes direction significantly
- Requires 2-frame confirmation: ball must change direction 140° turn

#### C. **Motion Constraints**

All shots require:
- **Walking Check:** Player must NOT be walking
- **Nearest Ball:** Uses closest ball to player
- **Ball Liveness:** Ball must be in recent frames (last 1 second)

**Key Files:**
- `scripts/classification.py` - `ShotDetectionEngine` class
- `scripts/classification.py` - `ShotCandidateTracker` class

---

### Task 5: Motion & Movement Analysis

**What it does:** Tracks player movements to help distinguish between walking and swinging.

**How it works:**

1. **Walking Detection:**
   - Tracks shoulder and leg movements
   - Calculates velocity of shoulders and hips
   - Detects alternating leg motion (sign changes)
   - If enough signs of walking → player is walking
   - Prevents false shot detections during movement

2. **Pose History:**
   - Keeps last N frames of player's pose
   - Allows calculating velocity of body parts
   - Used to detect sustained movement vs quick swings

3. **Anchor Point Tracking:**
   - Uses center of hip area as player's anchor point
   - Tracks displacement of anchor
   - Measures if player is stationary (good for serves)

4. **Movement Tracker:**
   - Records left/right wrist positions over time
   - Calculates wrist velocity vectors
   - Identifies which wrist is dominant

**Key File:** `scripts/classification.py` - `SkeletonMovementTracker`, `PoseHistoryTracker`

---

## Processing Pipeline

### Frame-by-Frame Processing

For each frame in the video:

```python
# Step 1: Run YOLO models
persons = detect_persons(frame)          # YOLOv8 detector
balls = detect_balls(frame)              # Ball-specific YOLOv8
poses = detect_poses(persons, frame)     # YOLOv8-Pose

# Step 2: Update trackers
ball_tracker.update(balls, frame_index)  # Track ball across frames
person_tracker.update(persons)           # Track people across frames

# Step 3: Process each player
for person in persons:
    player_data = analyze_player(person, poses, ball_tracker)
    
# Step 4: Detect shots
shot_events = shot_engine.process_frame(
    frame_index, player_data, pose_history, ball_tracker
)

# Step 5: Draw and save
draw_annotations(frame, persons, shots, ball_tracker)
output_video.write(frame)
```

### Output Generation

After processing all frames:
- **shots.jsonl** - One JSON object per line, each containing detected shot data
- **output_video.mp4** - Annotated video with boxes, IDs, shot labels

---

## Output Format

The `shots.jsonl` file contains one shot per line:

```json
{
  "player_id": 1,
  "frame": 453,
  "second": 15.1,
  "shot": "forehand"
}
```

**Fields:**
- `player_id`: Unique identifier for the player (assigned by PersonTracker)
- `frame`: Frame number where shot occurred (0-indexed)
- `second`: Time in seconds from video start
- `shot`: Type of shot - "forehand", "backhand", or "serve"

---

## Key Concepts

### Bounding Box Format
- `(x1, y1, x2, y2)` = top-left corner (x1,y1) and bottom-right corner (x2,y2)
- Units: pixel coordinates in video frame

### Confidence Score
- Range: 0.0 to 1.0
- Higher = more confident detection
- Detections with low confidence may be false positives

### Track vs Detection
- **Detection:** Finding something in one frame
- **Track:** Following the same object across multiple frames

### Interpolation
- When ball is briefly not detected, system predicts its position
- Uses velocity from visible frames to extrapolate
- Essential for handling occasional detection failures

### Dead Ball vs Live Ball
- **Dead Ball:** Stationary or barely moving (not in play)
- **Live Ball:** Moving significantly (in play)
- Helps distinguish between serves and actual game play

---

## Configuration

Key parameters in `scripts/utils.py`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `PIXELS_PER_METER` | 100 | Conversion factor for speed estimation |
| `WRIST_CLOSE_PX` | 4 | Distance threshold for backhand detection |
| `SWING_WRIST_VEL_PX` | 18 | Minimum wrist speed to detect swing |
| `SWING_DIR_ANGLE_DEG` | 60 | Wrist-to-ball angle threshold |
| `SHOT_CONFIRM_TURN_DEG` | 140 | Ball direction change for forehand confirm |
| `SHOT_CONFIRM_SEC` | 0.5 | Time window for shot confirmation |
| `BALL_MOVE_MIN_PX` | 8 | Minimum ball movement for liveness |
| `PLAYER_STATIC_PX` | 12 | Anchor displacement threshold |
| `WALK_WINDOW_SEC` | 1.0 | Time window for walk detection |

---

## Summary

This system combines multiple AI models and custom logic to:
1. **Find people** in the court using object detection
2. **Track the ball** across frames using motion prediction
3. **Understand poses** using skeleton detection
4. **Identify shots** using movement patterns and physics rules
5. **Output results** in a simple, searchable format

The key innovation is combining skeleton data with ball tracking to accurately identify shots without requiring slow-motion or special hardware.
