"""Wide / No-ball detector for cricket video.

Pipeline:
  1. Detect the ball with the YOLOv8 cricket-ball model trained in this repo
     (runs/detect/train5/weights/best.pt).
  2. Detect the bowler's pose with YOLOv8-Pose (downloaded by Ultralytics on
     first run).
  3. Use a 4-point image -> pitch homography (clicked once during calibration)
     to reason about the popping creases, return creases and wide markers in
     real-world metres.
  4. Overlay WIDE / NO BALL alerts on the video when:
        * the ball crosses the batsman's popping crease outside the
          +/- 0.89 m wide markers, OR
        * the bowler's front foot lands past the bowler-end popping crease.

Usage
-----
Run interactive crease calibration (4 clicks, see prompt on screen):
    python wide_no_ball_detector.py --video videos/test1.mp4 --calibrate

Run detection with an existing calibration:
    python wide_no_ball_detector.py --video videos/test1.mp4 \\
        --calibration crease_config.json --output runs/wide_no_ball.mp4

Run headless (no GUI window) and write annotated MP4:
    python wide_no_ball_detector.py --video videos/test1.mp4 \\
        --calibration crease_config.json --output runs/wide_no_ball.mp4 \\
        --no-show
"""

from __future__ import annotations

import argparse
import json
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# --------------------------------------------------------------------------- #
# Pitch geometry (real-world metres).
# Origin: middle of the bowler-end popping crease.
# +y  : along the pitch toward the batsman.
# +x  : to the right when looking from the bowler toward the batsman.
# --------------------------------------------------------------------------- #
RETURN_CREASE_HALF_WIDTH = 0.66   # 4 ft 4 in apart -> +/- 0.66 m
WIDE_MARKER_HALF_WIDTH = 0.89     # 35 in from middle stump -> +/- 0.89 m
PITCH_LENGTH = 17.68              # popping crease to popping crease

# Order the user clicks during calibration:
#   1) bowler-end popping crease  x  LEFT  return crease
#   2) bowler-end popping crease  x  RIGHT return crease
#   3) batsman-end popping crease x  RIGHT return crease
#   4) batsman-end popping crease x  LEFT  return crease
PITCH_REF_POINTS = np.float32([
    [-RETURN_CREASE_HALF_WIDTH, 0.0],
    [ RETURN_CREASE_HALF_WIDTH, 0.0],
    [ RETURN_CREASE_HALF_WIDTH, PITCH_LENGTH],
    [-RETURN_CREASE_HALF_WIDTH, PITCH_LENGTH],
])

CALIBRATION_INSTRUCTIONS = [
    "1) Bowler end popping crease  -  LEFT  return crease",
    "2) Bowler end popping crease  -  RIGHT return crease",
    "3) Batsman end popping crease -  RIGHT return crease",
    "4) Batsman end popping crease -  LEFT  return crease",
]


# --------------------------------------------------------------------------- #
# Homography helpers
# --------------------------------------------------------------------------- #
def compute_homography(image_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H, _ = cv2.findHomography(image_points, PITCH_REF_POINTS)
    if H is None:
        raise RuntimeError("findHomography failed - check your 4 points.")
    H_inv = np.linalg.inv(H)
    return H, H_inv


def img_to_pitch(H: np.ndarray, points) -> np.ndarray:
    pts = np.array(points, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)


def pitch_to_img(H_inv: np.ndarray, points) -> np.ndarray:
    pts = np.array(points, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H_inv)
    return out.reshape(-1, 2).astype(int)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def calibrate_homography(video_path: str, save_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise SystemExit(f"Could not read first frame from {video_path}")

    clicked: list[tuple[int, int]] = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
            clicked.append((x, y))

    cv2.namedWindow("calibration")
    cv2.setMouseCallback("calibration", on_mouse)

    while True:
        canvas = frame.copy()
        for i, p in enumerate(clicked):
            cv2.circle(canvas, p, 6, (0, 255, 0), -1)
            cv2.putText(canvas, str(i + 1), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(clicked) >= 2:
            cv2.line(canvas, clicked[0], clicked[1], (0, 255, 0), 2)
        if len(clicked) == 4:
            cv2.line(canvas, clicked[2], clicked[3], (0, 255, 0), 2)
            cv2.line(canvas, clicked[1], clicked[2], (0, 200, 200), 1)
            cv2.line(canvas, clicked[3], clicked[0], (0, 200, 200), 1)

        if len(clicked) < 4:
            prompt = f"Click point {len(clicked) + 1} of 4"
        else:
            prompt = "ENTER = save   r = redo   q = quit"
        cv2.putText(canvas, prompt, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        for i, txt in enumerate(CALIBRATION_INSTRUCTIONS):
            color = (0, 255, 0) if i < len(clicked) else (200, 200, 200)
            cv2.putText(canvas, txt, (10, 70 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        cv2.imshow("calibration", canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            raise SystemExit("Calibration cancelled.")
        if key == ord('r'):
            clicked.clear()
        if key in (13, 10) and len(clicked) == 4:
            break

    cv2.destroyAllWindows()
    save_calibration(np.float32(clicked), save_path)


def save_calibration(image_points: np.ndarray, save_path: str) -> None:
    H, H_inv = compute_homography(image_points)
    data = {
        "image_points": image_points.tolist(),
        "pitch_points": PITCH_REF_POINTS.tolist(),
        "homography": H.tolist(),
        "homography_inv": H_inv.tolist(),
    }
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[calibration] saved to {save_path}")


def load_calibration(path: str):
    with open(path) as f:
        data = json.load(f)
    H = np.array(data["homography"], dtype=np.float64)
    H_inv = np.array(data["homography_inv"], dtype=np.float64)
    image_points = np.array(data["image_points"], dtype=np.float32)
    return H, H_inv, image_points


# --------------------------------------------------------------------------- #
# Pitch overlay
# --------------------------------------------------------------------------- #
def draw_pitch_overlay(frame: np.ndarray, H_inv: np.ndarray) -> None:
    extend = WIDE_MARKER_HALF_WIDTH + 0.3
    bowler_pop = pitch_to_img(H_inv, [(-extend, 0.0), (extend, 0.0)])
    batsman_pop = pitch_to_img(
        H_inv, [(-extend, PITCH_LENGTH), (extend, PITCH_LENGTH)])
    cv2.line(frame, tuple(bowler_pop[0]), tuple(bowler_pop[1]),
             (0, 200, 255), 2)
    cv2.line(frame, tuple(batsman_pop[0]), tuple(batsman_pop[1]),
             (0, 200, 255), 2)

    for sign in (-1, 1):
        x = sign * RETURN_CREASE_HALF_WIDTH
        ret = pitch_to_img(H_inv,
                           [(x, -0.4), (x, PITCH_LENGTH + 0.4)])
        cv2.line(frame, tuple(ret[0]), tuple(ret[1]), (170, 170, 170), 1)

    # Wide markings: short perpendicular ticks at the batter's popping
    # crease at +/- 0.89 m from middle stump, exactly as they are painted on
    # the pitch.  The dashed segment between them visualises the legal
    # "channel" - any ball passing the batter outside this channel is wide.
    left_tick = pitch_to_img(
        H_inv,
        [(-WIDE_MARKER_HALF_WIDTH, PITCH_LENGTH - 0.6),
         (-WIDE_MARKER_HALF_WIDTH, PITCH_LENGTH + 0.4)])
    right_tick = pitch_to_img(
        H_inv,
        [(WIDE_MARKER_HALF_WIDTH, PITCH_LENGTH - 0.6),
         (WIDE_MARKER_HALF_WIDTH, PITCH_LENGTH + 0.4)])
    cv2.line(frame, tuple(left_tick[0]), tuple(left_tick[1]),
             (0, 0, 255), 4)
    cv2.line(frame, tuple(right_tick[0]), tuple(right_tick[1]),
             (0, 0, 255), 4)

    # Dashed connector along the popping crease between the two ticks.
    n_dash = 20
    for i in range(n_dash):
        if i % 2 == 0:
            x0 = -WIDE_MARKER_HALF_WIDTH + (2 * WIDE_MARKER_HALF_WIDTH) * i / n_dash
            x1 = -WIDE_MARKER_HALF_WIDTH + (2 * WIDE_MARKER_HALF_WIDTH) * (i + 1) / n_dash
            seg = pitch_to_img(H_inv,
                               [(x0, PITCH_LENGTH), (x1, PITCH_LENGTH)])
            cv2.line(frame, tuple(seg[0]), tuple(seg[1]), (0, 0, 255), 1)

    label = pitch_to_img(
        H_inv, [(WIDE_MARKER_HALF_WIDTH + 0.05, PITCH_LENGTH + 0.2)])[0]
    cv2.putText(frame, "wide line", (label[0] + 4, label[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)


# --------------------------------------------------------------------------- #
# Alert banner
# --------------------------------------------------------------------------- #
class Alert:
    def __init__(self) -> None:
        self.text: str | None = None
        self.color = (255, 255, 255)
        self.frames_left = 0

    def trigger(self, text: str, color: tuple[int, int, int],
                hold_frames: int = 60) -> None:
        if self.text != text or self.frames_left == 0:
            print(f"[ALERT] {text}")
        self.text = text
        self.color = color
        self.frames_left = max(self.frames_left, hold_frames)

    def draw(self, frame: np.ndarray) -> None:
        if self.frames_left <= 0 or self.text is None:
            return
        h, w = frame.shape[:2]
        box_h = max(70, h // 8)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, box_h), self.color, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        (tw, th), _ = cv2.getTextSize(self.text,
                                      cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
        cv2.putText(frame, self.text,
                    ((w - tw) // 2, (box_h + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
        self.frames_left -= 1


# --------------------------------------------------------------------------- #
# Detection loop
# --------------------------------------------------------------------------- #
def pick_bowler_front_foot(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    H: np.ndarray,
):
    """Return (image_xy, pitch_xy) of the bowler's most-forward ankle.

    The bowler is heuristically the detected person whose ankle is closest to
    the bowler-end popping crease (pitch y = 0) while remaining inside a
    +/- 3 m lateral lane around the pitch.
    """
    best_score = float("inf")
    best_front_img = None
    best_front_pitch = None

    n_people = kpts_xy.shape[0]
    for i in range(n_people):
        l_ankle = kpts_xy[i, 15]
        r_ankle = kpts_xy[i, 16]
        l_conf = kpts_conf[i, 15] if kpts_conf is not None else 1.0
        r_conf = kpts_conf[i, 16] if kpts_conf is not None else 1.0

        ankles_img = []
        if l_conf > 0.3 and not np.allclose(l_ankle, 0):
            ankles_img.append(l_ankle)
        if r_conf > 0.3 and not np.allclose(r_ankle, 0):
            ankles_img.append(r_ankle)
        if not ankles_img:
            continue

        ankles_img_arr = np.array(ankles_img, dtype=np.float64)
        ankles_pitch = img_to_pitch(H, ankles_img_arr)

        # The bowler delivers from near pitch-y = 0 and within the lane.
        in_lane = np.abs(ankles_pitch[:, 0]) < 3.0
        if not np.any(in_lane):
            continue
        # Score = how close any of this person's feet are to the bowler-end
        # crease.
        ys = ankles_pitch[in_lane, 1]
        score = np.min(np.abs(ys))
        # Reject people who are clearly not the bowler:
        # - closest ankle more than 3 m past the popping crease   -> striker
        # - all ankles entirely behind the bowling crease (~-1.5 m) -> umpire
        # - feet far outside the pitch laterally                   -> fielder
        if np.min(ys) > 3.0:
            continue
        if np.max(ys) < -1.5:
            continue
        if np.min(np.abs(ankles_pitch[:, 0])) > 2.0:
            continue

        if score < best_score:
            best_score = score
            # The "front" foot for a bowler is the one furthest forward in
            # pitch coords (largest y).
            fwd_idx = int(np.argmax(ankles_pitch[:, 1]))
            best_front_img = ankles_img_arr[fwd_idx]
            best_front_pitch = ankles_pitch[fwd_idx]

    return best_front_img, best_front_pitch


def run_detection(
    video_path: str,
    calibration_path: str,
    output_path: str | None,
    ball_weights: str,
    pose_weights: str,
    conf: float,
    show: bool,
    max_frames: int | None,
) -> dict:
    H, H_inv, _ = load_calibration(calibration_path)

    print(f"[load] ball weights:  {ball_weights}")
    ball_model = YOLO(ball_weights)
    print(f"[load] pose weights:  {pose_weights}")
    pose_model = YOLO(pose_weights)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {w}x{h} @ {fps:.1f} fps - {total} frames")

    writer = None
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        print(f"[video] writing annotated output to {output_path}")

    ball_img_hist: deque[tuple[int, int]] = deque(maxlen=40)
    ball_pitch_hist: deque[np.ndarray] = deque(maxlen=40)
    alert = Alert()
    wide_called = False
    no_ball_called = False
    no_ball_streak = 0
    wide_streak = 0
    frame_idx = 0
    stats = {"frames": 0, "ball_seen": 0, "bowler_seen": 0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        stats["frames"] += 1
        if max_frames is not None and frame_idx > max_frames:
            break

        draw_pitch_overlay(frame, H_inv)

        # ---- Ball ----
        ball_res = ball_model.predict(frame, conf=conf, verbose=False)[0]
        ball_xy = None
        if ball_res.boxes is not None and len(ball_res.boxes) > 0:
            confs = ball_res.boxes.conf.cpu().numpy()
            xyxy = ball_res.boxes.xyxy.cpu().numpy()
            idx = int(np.argmax(confs))
            x1, y1, x2, y2 = xyxy[idx]
            ball_xy = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 0, 255), 2)
            cv2.circle(frame, ball_xy, 4, (0, 0, 255), -1)
            stats["ball_seen"] += 1

        if ball_xy is not None:
            ball_img_hist.append(ball_xy)
            ball_pitch_hist.append(img_to_pitch(H, [ball_xy])[0])

        # Ball trail
        for i in range(1, len(ball_img_hist)):
            cv2.line(frame, ball_img_hist[i - 1], ball_img_hist[i],
                     (255, 0, 0), 3)

        # ---- Pose ----
        pose_res = pose_model.predict(frame, conf=0.4, verbose=False)[0]
        front_foot_img = None
        front_foot_pitch = None
        if (pose_res.keypoints is not None
                and pose_res.keypoints.xy is not None
                and len(pose_res.keypoints.xy) > 0):
            kpts_xy = pose_res.keypoints.xy.cpu().numpy()  # (N, 17, 2)
            kpts_conf = (pose_res.keypoints.conf.cpu().numpy()
                         if pose_res.keypoints.conf is not None else None)
            front_foot_img, front_foot_pitch = pick_bowler_front_foot(
                kpts_xy, kpts_conf, H)
            if front_foot_img is not None:
                stats["bowler_seen"] += 1
                cv2.circle(frame,
                           (int(front_foot_img[0]), int(front_foot_img[1])),
                           8, (0, 255, 255), 2)

        # ---- No-ball logic ----
        if front_foot_pitch is not None:
            px, py = front_foot_pitch
            cv2.putText(
                frame,
                f"front foot  pitch=({px:+.2f},{py:+.2f}) m",
                (12, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 1)
            # The bowler's front foot is illegal if it lands clearly past the
            # popping crease (origin at y = 0) while still on the pitch.
            # Tolerance is generous (15 cm) because the pose keypoint is at
            # the ankle joint, not the heel - and a small calibration error
            # can easily produce a few cm of drift.  Require persistence
            # over several frames to ignore pose noise.
            if py > 0.15 and abs(px) < 1.5:
                no_ball_streak += 1
            else:
                no_ball_streak = max(0, no_ball_streak - 1)
            if no_ball_streak >= 3 and not no_ball_called:
                no_ball_called = True
                alert.trigger("NO BALL", (0, 0, 200), hold_frames=int(fps * 2))
            if no_ball_called and front_foot_img is not None:
                cv2.circle(frame,
                           (int(front_foot_img[0]), int(front_foot_img[1])),
                           14, (0, 0, 255), 3)
        else:
            no_ball_streak = 0

        # ---- Wide logic ----
        # The wide markers sit at the batter's popping crease at x = +/-
        # 0.89 m.  Fire WIDE when the ball reaches the batter (close to the
        # popping crease) while outside that channel.  We accept any ball
        # detection inside a +/- 1.5 m neighbourhood of the batter's crease
        # so a single missed frame at the exact crossing does not cost the
        # call.
        if not wide_called and len(ball_pitch_hist) >= 1:
            curr = ball_pitch_hist[-1]
            cx, cy = float(curr[0]), float(curr[1])
            near_batter = (PITCH_LENGTH - 1.5 < cy < PITCH_LENGTH + 2.0)
            outside_wide = abs(cx) > WIDE_MARKER_HALF_WIDTH

            cv2.putText(
                frame,
                f"ball pitch  x={cx:+.2f} m  y={cy:+.2f} m",
                (12, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

            if near_batter and outside_wide:
                wide_streak += 1
            else:
                wide_streak = max(0, wide_streak - 1)
            if wide_streak >= 2:
                wide_called = True
                alert.trigger("WIDE", (0, 140, 255),
                              hold_frames=int(fps * 2))
                bx, by = ball_img_hist[-1]
                cv2.circle(frame, (bx, by), 14, (0, 0, 255), 3)

        alert.draw(frame)

        cv2.putText(frame, "WIDE / NO-BALL DETECTOR",
                    (12, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)

        if writer is not None:
            writer.write(frame)
        if show:
            cv2.imshow("wide-no-ball",
                       cv2.resize(frame, (min(1100, w * 2), min(620, h * 2))))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print("\n[summary]")
    print(f"  frames processed : {stats['frames']}")
    print(f"  ball detections  : {stats['ball_seen']}")
    print(f"  bowler detected  : {stats['bowler_seen']}")
    print(f"  WIDE triggered   : {wide_called}")
    print(f"  NO BALL triggered: {no_ball_called}")
    return {
        "frames": stats["frames"],
        "ball_detections": stats["ball_seen"],
        "bowler_detections": stats["bowler_seen"],
        "wide": wide_called,
        "no_ball": no_ball_called,
        "video_size": [w, h],
        "fps": fps,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect wides and no-balls in a cricket video.")
    parser.add_argument("--video", required=True,
                        help="Path to the input video.")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run interactive 4-click crease calibration.")
    parser.add_argument("--calibration", default="crease_config.json",
                        help="Calibration JSON path (read or written).")
    parser.add_argument("--output", default=None,
                        help="Optional MP4 path for annotated output.")
    parser.add_argument(
        "--ball-weights",
        default=os.path.join("runs", "detect", "train5", "weights", "best.pt"),
        help="YOLOv8 cricket-ball detector weights.")
    parser.add_argument("--pose-weights", default="yolov8n-pose.pt",
                        help="YOLOv8-Pose weights (auto-downloaded).")
    parser.add_argument("--conf", type=float, default=0.30,
                        help="Ball detection confidence threshold.")
    parser.add_argument("--no-show", action="store_true",
                        help="Disable the live preview window.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional cap on frames processed.")
    args = parser.parse_args()

    if args.calibrate:
        calibrate_homography(args.video, args.calibration)
        return

    if not Path(args.calibration).is_file():
        raise SystemExit(
            f"No calibration at {args.calibration}. "
            f"Re-run with --calibrate to create one.")

    run_detection(
        video_path=args.video,
        calibration_path=args.calibration,
        output_path=args.output,
        ball_weights=args.ball_weights,
        pose_weights=args.pose_weights,
        conf=args.conf,
        show=not args.no_show,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
