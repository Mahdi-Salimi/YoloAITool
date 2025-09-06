from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
import torch
import os
import cv2
from utils import get_best_frame
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import requests

# Load .env
load_dotenv()
BASE_VIDEO_PATH = os.getenv("BASE_VIDEO_PATH", "/videos")

# Logging setup
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("yolo_service")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler("logs/service.log", maxBytes=5*1024*1024, backupCount=2)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# FastAPI app
app = FastAPI()

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILE = "yolov8n.pt"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
MODEL_URL = "https://huggingface.co/ultralytics/yolov8/resolve/main/yolov8n.pt"

# Download model to models/ if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print(f"Downloading YOLO model to {MODEL_PATH} ...")
    r = requests.get(MODEL_URL, stream=True)
    if r.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("✅ YOLO model downloaded successfully")
    else:
        raise RuntimeError(f"Failed to download YOLO model (status {r.status_code})")

# Set device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model from models directory
model = YOLO(MODEL_PATH).to(device)
print(f"Loaded YOLO model on {device}, stored at {MODEL_PATH}")

# Pydantic request model
class VideoRequest(BaseModel):
    relative_path: str        # e.g., "ugc/video1.mp4"
    callback_url: str         # URL to call after processing

@app.post("/generate-thumbnail/")
async def generate_thumbnail(request: VideoRequest):
    video_path = os.path.join(BASE_VIDEO_PATH, request.relative_path)
    if not os.path.exists(video_path):
        logger.error(f"Video not found: {video_path}")
        payload = {"status": "fail", "description": f"Video not found: {video_path}"}
        try:
            requests.post(request.callback_url, json=payload)
        except Exception as e:
            logger.error(f"Callback failed: {e}")
        return payload

    logger.info(f"Processing video: {video_path}")
    try:
        best_frame = get_best_frame(video_path, model)
        thumbnail_path = os.path.join("/tmp", f"thumbnail_{os.path.basename(video_path)}.jpg")
        cv2.imwrite(thumbnail_path, best_frame)
        logger.info(f"Thumbnail saved: {thumbnail_path}")

        payload = {"status": "success", "thumbnail_path": thumbnail_path}
        try:
            requests.post(request.callback_url, json=payload)
        except Exception as e:
            logger.error(f"Callback failed: {e}")
        return payload

    except Exception as e:
        logger.error(f"Error processing {video_path}: {e}")
        payload = {"status": "fail", "description": str(e)}
        try:
            requests.post(request.callback_url, json=payload)
        except Exception as e2:
            logger.error(f"Callback failed: {e2}")
        return payload
