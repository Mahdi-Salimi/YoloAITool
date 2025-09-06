import cv2
import numpy as np

def extract_frames(video_path, frame_step=1):
    """Extract frames every Nth frame from video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_ids = []
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_step == 0:
            frames.append(frame)
            frame_ids.append(frame_id)
        frame_id += 1
    cap.release()
    return frames, frame_ids


def score_frame(frame, model):
    """Score a frame based on face detection + sharpness + brightness."""
    results = model(frame)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return 0

    scores = []
    for box in boxes:
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)

        # Weighted scoring
        score = conf * 0.5 + (sharpness / 100.0) * 0.3 + (brightness / 255.0) * 0.2
        scores.append(score)

    return max(scores) if scores else 0


def get_best_frame(video_path, model, frame_step=2):
    """Extract frames, score them, and return the best frame for thumbnail."""
    frames, frame_ids = extract_frames(video_path, frame_step)
    best_score = -1
    best_frame = None
    best_id = None

    for frame, frame_id in zip(frames, frame_ids):
        score = score_frame(frame, model)
        if score > best_score:
            best_score = score
            best_frame = frame
            best_id = frame_id

    return best_frame
