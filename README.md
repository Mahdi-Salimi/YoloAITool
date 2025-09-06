# YOLO Thumbnail Service 

FastAPI microservice that generates the best thumbnail from UGC videos using YOLOv8 face detection.

---

## Features

- Accepts **relative video path** and **callback_url**
- Constructs absolute path using `.env BASE_VIDEO_PATH`
- GPU-aware: uses GPU if available, CPU fallback
- Logging to `logs/service.log` with rotation
- Dockerized with Docker Compose
- After processing, sends POST request to **callback_url**:
  - Success: `{status: "success", thumbnail_path: "..."}`  
  - Fail: `{status: "fail", description: "error reason"}`


