# Backend container for the wide / no-ball detector.
# Targets Render's free Docker tier (also works on Fly.io, Railway, HF Spaces).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV runtime + ffmpeg for H.264 transcoding of annotated MP4s.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache-friendly layer).
# We pin a slim, CPU-only stack and skip the heavyweight pieces of the
# original kushagra3204 requirements.txt (tensorflow, labelImg, PyQt, etc.)
# that the runtime does not need.
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# App source.  .dockerignore strips the unused YOLOv8 sizes (l/m/s/n)
# and the user-data dirs.
COPY . .

ENV HOST=0.0.0.0 \
    PORT=8080
EXPOSE 8080

# 1 worker (each loads YOLO into memory ~1 GB) + long timeout (CPU detection).
CMD ["gunicorn", "-b", "0.0.0.0:8080", "-w", "1", "--threads", "2", "--timeout", "600", "app:app"]
