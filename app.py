"""Flask UI for the wide / no-ball detector.

Workflow exposed to the browser:

    1.  POST /api/upload          - upload an MP4
    2.  GET  /api/job/<id>/frame  - first frame as a PNG (for calibration)
    3.  POST /api/job/<id>/calibrate
                                  - 4 image-space crease points (JSON)
    4.  POST /api/job/<id>/detect - run detection, return result JSON
    5.  GET  /uploads/<file>      - serve raw upload
    6.  GET  /outputs/<file>      - serve annotated MP4 (H.264, browser-safe)

The detection itself reuses `wide_no_ball_detector.run_detection`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import (Flask, jsonify, render_template, request, send_from_directory,
                   url_for)
try:
    from flask_cors import CORS
except ImportError:  # local dev without flask-cors installed
    CORS = None

from wide_no_ball_detector import (
    run_detection,
    save_calibration,
)

# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
CALIB_DIR = BASE_DIR / "calibrations"
for d in (UPLOAD_DIR, OUTPUT_DIR, CALIB_DIR):
    d.mkdir(exist_ok=True)

BALL_WEIGHTS = str(BASE_DIR / "runs" / "detect" / "train5" / "weights" /
                   "best.pt")
POSE_WEIGHTS = str(BASE_DIR / "yolov8n-pose.pt")
MAX_UPLOAD_MB = 200
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Allow the Vercel-hosted static frontend (and any other origin you set in
# CORS_ALLOW_ORIGINS, comma-separated) to call this API.
if CORS is not None:
    _cors_origins = os.environ.get("CORS_ALLOW_ORIGINS", "*")
    _origins = [o.strip() for o in _cors_origins.split(",") if o.strip()]
    CORS(app, resources={r"/*": {"origins": _origins or "*"}})


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def job_paths(job_id: str) -> dict:
    return {
        "upload": UPLOAD_DIR / f"{job_id}.mp4",
        "frame": UPLOAD_DIR / f"{job_id}_frame.png",
        "calibration": CALIB_DIR / f"{job_id}.json",
        "output_raw": OUTPUT_DIR / f"{job_id}_raw.mp4",
        "output": OUTPUT_DIR / f"{job_id}.mp4",
        "meta": OUTPUT_DIR / f"{job_id}.json",
    }


def transcode_to_h264(src: Path, dst: Path) -> bool:
    """Re-encode src -> dst as H.264 so browsers (Chrome/Edge/Firefox) can
    play the annotated MP4 inline.  Returns True on success."""
    if not shutil.which("ffmpeg"):
        return False
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", str(src),
             "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart",
             "-an",
             str(dst)],
            check=True,
        )
        return dst.is_file() and dst.stat().st_size > 0
    except Exception as exc:  # pragma: no cover - defensive
        app.logger.warning("ffmpeg transcode failed: %s", exc)
        return False


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def api_upload():
    file = request.files.get("video")
    if file is None or not file.filename:
        return jsonify({"error": "no file uploaded"}), 400
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"unsupported extension {ext}"}), 400

    job_id = uuid.uuid4().hex[:12]
    paths = job_paths(job_id)
    file.save(paths["upload"])

    cap = cv2.VideoCapture(str(paths["upload"]))
    if not cap.isOpened():
        paths["upload"].unlink(missing_ok=True)
        return jsonify({"error": "could not open video"}), 400
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return jsonify({"error": "could not read first frame"}), 400
    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    cv2.imwrite(str(paths["frame"]), frame)

    return jsonify({
        "job_id": job_id,
        "width": w,
        "height": h,
        "fps": fps,
        "frames": n_frames,
        "frame_url": url_for("api_frame", job_id=job_id),
        "video_url": url_for("serve_upload", filename=f"{job_id}.mp4"),
    })


@app.route("/api/job/<job_id>/frame")
def api_frame(job_id: str):
    paths = job_paths(job_id)
    if not paths["frame"].is_file():
        return jsonify({"error": "unknown job"}), 404
    return send_from_directory(UPLOAD_DIR, paths["frame"].name)


@app.route("/api/job/<job_id>/calibrate", methods=["POST"])
def api_calibrate(job_id: str):
    paths = job_paths(job_id)
    if not paths["upload"].is_file():
        return jsonify({"error": "unknown job"}), 404
    data = request.get_json(silent=True) or {}
    pts = data.get("points")
    if (not isinstance(pts, list)
            or len(pts) != 4
            or not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in pts)):
        return jsonify({"error": "expected 4 [x, y] points"}), 400
    image_points = np.float32([[float(x), float(y)] for x, y in pts])
    save_calibration(image_points, str(paths["calibration"]))
    return jsonify({"ok": True})


@app.route("/api/job/<job_id>/detect", methods=["POST"])
def api_detect(job_id: str):
    paths = job_paths(job_id)
    if not paths["upload"].is_file():
        return jsonify({"error": "unknown job"}), 404
    if not paths["calibration"].is_file():
        return jsonify({"error": "video is not calibrated"}), 400

    try:
        result = run_detection(
            video_path=str(paths["upload"]),
            calibration_path=str(paths["calibration"]),
            output_path=str(paths["output_raw"]),
            ball_weights=BALL_WEIGHTS,
            pose_weights=POSE_WEIGHTS,
            conf=0.30,
            show=False,
            max_frames=None,
        )
    except Exception as exc:
        app.logger.exception("detection failed")
        return jsonify({"error": f"detection failed: {exc}"}), 500

    # Re-encode the OpenCV mp4v output to browser-friendly H.264.
    final = paths["output"]
    if transcode_to_h264(paths["output_raw"], final):
        paths["output_raw"].unlink(missing_ok=True)
        served = final.name
    else:
        # Fallback: just rename and hope the browser plays mp4v.
        shutil.move(paths["output_raw"], final)
        served = final.name

    result["output_url"] = url_for("serve_output", filename=served)
    with open(paths["meta"], "w") as f:
        json.dump(result, f, indent=2)
    return jsonify(result)


@app.route("/uploads/<path:filename>")
def serve_upload(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/outputs/<path:filename>")
def serve_output(filename: str):
    return send_from_directory(OUTPUT_DIR, filename)


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    print(f"\n  open your browser at  http://{host}:{port}\n")
    app.run(host=host, port=port, debug=False)
