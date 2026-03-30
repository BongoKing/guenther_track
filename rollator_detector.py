"""
Rollator Detector for Augsburg Webcams
Periodically fetches webcam images and uses either Claude Vision (API) or a
local YOLOv8 model (free, offline) to detect rollators / walking frames.
Sends a Signal notification (with image) when a rollator is detected.
Logs all activity to rollator_detector.log and prints to console (detections in red).

Usage:
    # --- Using Claude (default) ---
    pip install anthropic requests
    python rollator_detector.py --engine claude

    # --- Using local YOLO (free, no API key needed) ---
    pip install ultralytics requests Pillow opencv-python numpy
    python rollator_detector.py --engine yolo

    # --- Live preview window with bounding boxes ---
    python rollator_detector.py --engine yolo --show

    # --- Common options ---
    python rollator_detector.py --engine yolo --interval 60 --confidence 0.3

Environment variables (Claude mode only):
    ANTHROPIC_API_KEY   your Anthropic key
Signal (optional, both modes):
    SIGNAL_SENDER       your registered signal-cli number
    SIGNAL_RECIPIENT    who to notify

Requires signal-cli for notifications:
    https://github.com/AsamK/signal-cli
"""

import argparse
import base64
import io
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

CAMERAS = {
    "Rathausplatz (Dachspitz)": (
        "https://www.augsburg.de/fileadmin/user_upload/header/webcam/webcamdachspitz/"
        "B_Rathausplatz_Dachspitz_00.jpg"
    ),
    "Rathaus & Perlachturm": (
        "https://www.augsburg.de/fileadmin/user_upload/header/webcam/webcamerker/"
        "B_Rathaus_und_Perlachturm_00.jpg"
    ),
}

CHECK_INTERVAL = 30  # default seconds between checks
CLAUDE_MODEL = "claude-sonnet-4-6"
SAVE_DIR = Path("rollator_detections")
LOG_FILE = Path("rollator_detector.log")

DETECTION_PROMPT = """Analyze this webcam image carefully.
Is there anyone using a rollator (a wheeled walker / walking frame) visible in the image?

Respond with JSON only, no other text:
{
  "rollator_detected": true/false,
  "confidence": "high"/"medium"/"low",
  "description": "brief description of what you see"
}"""

# YOLO class IDs that hint at a rollator.  COCO does not have a dedicated
# "rollator" class, so we look for *person* near objects that often co-occur
# with rollators: suitcase (28, similar boxy shape), handbag (26), backpack (24),
# chair (56, sometimes misclassified), bench (13).  A rollator by itself is
# most often classified as "suitcase", "chair", or occasionally "bicycle".
YOLO_ROLLATOR_HINT_CLASSES = {13, 24, 26, 28, 56}  # bench, backpack, handbag, suitcase, chair
YOLO_PERSON_CLASS = 0

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

# Bounding box colors (BGR for OpenCV)
COLOR_PERSON = (255, 200, 0)       # cyan-ish
COLOR_HINT = (0, 200, 255)         # orange
COLOR_ROLLATOR = (0, 0, 255)       # red — rollator match
COLOR_OTHER = (200, 200, 200)      # grey
COLOR_CLAUDE_OK = (0, 200, 0)      # green
COLOR_CLAUDE_DETECT = (0, 0, 255)  # red

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("rollator_detector")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)


class ColorConsoleHandler(logging.StreamHandler):
    """Console handler that colors detections red and no-rollator green."""

    def emit(self, record):
        if record.levelno >= logging.CRITICAL:
            record.msg = f"{RED}{record.msg}{RESET}"
        elif record.levelno == logging.WARNING:
            record.msg = f"{GREEN}{record.msg}{RESET}"
        super().emit(record)


console_handler = ColorConsoleHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(console_handler)


def log_info(msg: str):
    logger.info(msg)


def log_error(msg: str):
    logger.error(msg)


def log_clear(msg: str):
    """Log a no-rollator result — shown in green on console."""
    logger.warning(msg)


def log_detection(msg: str):
    """Log a rollator detection — shown in red on console, CRITICAL in log file."""
    logger.critical(msg)


# ---------------------------------------------------------------------------
# Image fetching
# ---------------------------------------------------------------------------

def fetch_image(url: str) -> bytes | None:
    """Fetch a webcam image, appending a timestamp to bypass caching."""
    try:
        fresh_url = f"{url}?{int(time.time() * 1000)}"
        resp = requests.get(fresh_url, timeout=10)
        resp.raise_for_status()
        return resp.content
    except requests.RequestException as e:
        log_error(f"  [ERROR] Failed to fetch image: {e}")
        return None


def bytes_to_cv2(image_data: bytes):
    """Convert raw image bytes to an OpenCV BGR numpy array."""
    import cv2
    arr = np.frombuffer(image_data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Engine: Claude Vision
# ---------------------------------------------------------------------------

def analyze_image_claude(client, image_data: bytes) -> dict:
    """Send image to Claude Vision and parse the rollator detection result."""
    b64 = base64.standard_b64encode(image_data).decode("utf-8")

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": DETECTION_PROMPT},
                ],
            }
        ],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return json.loads(text)


# ---------------------------------------------------------------------------
# Engine: Local YOLOv8
# ---------------------------------------------------------------------------

def load_yolo_model():
    """Load the YOLOv8 model (downloads automatically on first run)."""
    from ultralytics import YOLO
    log_info("  Loading YOLOv8 model (first run downloads ~6 MB)...")
    model = YOLO("yolov8n.pt")  # nano model — fast & free
    return model


def analyze_image_yolo(model, image_data: bytes, min_confidence: float) -> dict:
    """
    Run YOLOv8 on the image and look for a rollator.

    Returns the standard result dict plus a 'boxes' list for --show mode.
    """
    from PIL import Image

    img = Image.open(io.BytesIO(image_data))
    results = model(img, verbose=False)[0]

    persons = []
    hints = []
    all_boxes = []  # every detected box for drawing

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < min_confidence:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        label = results.names[cls_id]
        entry = {"cls": cls_id, "label": label, "conf": conf,
                 "bbox": (x1, y1, x2, y2), "is_rollator_match": False}
        all_boxes.append(entry)
        if cls_id == YOLO_PERSON_CLASS:
            persons.append(entry)
        if cls_id in YOLO_ROLLATOR_HINT_CLASSES:
            hints.append(entry)

    # Check if any person is close to a rollator-hint object
    detected = False
    description_parts = []

    if persons and hints:
        for p in persons:
            px_center = (p["bbox"][0] + p["bbox"][2]) / 2
            py_bottom = p["bbox"][3]  # feet area
            for h in hints:
                hx_center = (h["bbox"][0] + h["bbox"][2]) / 2
                hy_center = (h["bbox"][1] + h["bbox"][3]) / 2
                dist_x = abs(px_center - hx_center)
                dist_y = abs(py_bottom - hy_center)
                if dist_x < 200 and dist_y < 200:
                    detected = True
                    p["is_rollator_match"] = True
                    h["is_rollator_match"] = True
                    description_parts.append(
                        f"person near '{h['label']}' (conf {h['conf']:.0%})"
                    )

    all_labels = [b["label"] for b in all_boxes]

    if detected:
        conf_str = "medium"
        desc = f"Possible rollator: {'; '.join(description_parts)}. All objects: {', '.join(all_labels)}"
    else:
        conf_str = "high" if not hints else "medium"
        desc = f"Objects detected: {', '.join(all_labels) if all_labels else 'none'}"

    return {
        "rollator_detected": detected,
        "confidence": conf_str,
        "description": desc,
        "boxes": all_boxes,
    }


# ---------------------------------------------------------------------------
# Live display (--show)
# ---------------------------------------------------------------------------

def draw_yolo_boxes(image_data: bytes, result: dict, camera_name: str):
    """Draw YOLO bounding boxes on a copy of the image. Returns a cv2 image."""
    import cv2

    img = bytes_to_cv2(image_data)
    if img is None:
        return None

    for box in result.get("boxes", []):
        x1, y1, x2, y2 = box["bbox"]
        label = f"{box['label']} {box['conf']:.0%}"

        if box["is_rollator_match"]:
            color = COLOR_ROLLATOR
            thickness = 3
        elif box["cls"] == YOLO_PERSON_CLASS:
            color = COLOR_PERSON
            thickness = 2
        elif box["cls"] in YOLO_ROLLATOR_HINT_CLASSES:
            color = COLOR_HINT
            thickness = 2
        else:
            color = COLOR_OTHER
            thickness = 1

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Camera name header
    detected = result.get("rollator_detected", False)
    header_color = COLOR_ROLLATOR if detected else (0, 180, 0)
    status = "ROLLATOR!" if detected else "clear"
    header = f"{camera_name}  [{status}]"
    cv2.putText(img, header, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, header_color, 2, cv2.LINE_AA)

    return img


def draw_claude_overlay(image_data: bytes, result: dict, camera_name: str):
    """Draw a status overlay for Claude results (no bounding boxes available)."""
    import cv2

    img = bytes_to_cv2(image_data)
    if img is None:
        return None

    detected = result.get("rollator_detected", False)
    confidence = result.get("confidence", "?")
    desc = result.get("description", "")

    # Header
    header_color = COLOR_CLAUDE_DETECT if detected else COLOR_CLAUDE_OK
    status = "ROLLATOR DETECTED!" if detected else "No rollator"
    header = f"{camera_name}  [{status}]  conf={confidence}"
    cv2.putText(img, header, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, header_color, 2, cv2.LINE_AA)

    # Description (word-wrapped at bottom)
    if desc:
        # Simple word wrap
        max_chars = max(40, img.shape[1] // 12)
        lines = [desc[i:i + max_chars] for i in range(0, len(desc), max_chars)]
        y = img.shape[0] - 10 - (len(lines) - 1) * 22
        for line in lines:
            # Dark background for readability
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (8, y - th - 4), (12 + tw, y + 4), (0, 0, 0), -1)
            cv2.putText(img, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22

    # Red border if detected
    if detected:
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), COLOR_CLAUDE_DETECT, 4)

    return img


_window_created = False


def show_combined(frames: dict, max_width: int = 1400, max_height: int = 800):
    """
    Display all camera frames side-by-side in a single OpenCV window,
    scaled to fit the screen.
    Returns True if the user pressed 'q' to quit.
    """
    global _window_created
    import cv2

    images = [f for f in frames.values() if f is not None]
    if not images:
        return False

    # Resize all to the same height
    target_h = min(img.shape[0] for img in images)
    resized = []
    for img in images:
        scale = target_h / img.shape[0]
        new_w = int(img.shape[1] * scale)
        resized.append(cv2.resize(img, (new_w, target_h)))

    # Add a thin separator
    sep = np.full((target_h, 3, 3), 80, dtype=np.uint8)
    parts = []
    for i, img in enumerate(resized):
        if i > 0:
            parts.append(sep)
        parts.append(img)

    combined = np.hstack(parts)

    # Timestamp bar at the bottom
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bar_h = 30
    bar = np.zeros((bar_h, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, f"Last update: {now_str}   |   Press Q to quit   |   ESC to close window",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    combined = np.vstack([combined, bar])

    # Scale down to fit screen
    h, w = combined.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        combined = cv2.resize(combined, (int(w * scale), int(h * scale)))

    # Create a resizable window (once)
    if not _window_created:
        cv2.namedWindow("Rollator Detector", cv2.WINDOW_NORMAL)
        disp_h, disp_w = combined.shape[:2]
        cv2.resizeWindow("Rollator Detector", disp_w, disp_h)
        _window_created = True

    cv2.imshow("Rollator Detector", combined)
    key = cv2.waitKey(1) & 0xFF
    return key in (ord("q"), ord("Q"), 27)  # q or ESC


# ---------------------------------------------------------------------------
# Image saving & Signal
# ---------------------------------------------------------------------------

def save_image(camera_name: str, image_data: bytes) -> Path:
    """Save a detected image to the detections directory."""
    SAVE_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = camera_name.replace(" ", "_").replace("&", "and")
    filename = SAVE_DIR / f"rollator_{safe_name}_{timestamp}.jpg"
    filename.write_bytes(image_data)
    return filename


def send_signal_message(
    sender: str, recipient: str, message: str, attachment: Path | None = None
):
    """Send a Signal message (with optional image attachment) via signal-cli."""
    cmd = ["signal-cli", "-a", sender, "send", "-m", message, recipient]
    if attachment and attachment.exists():
        cmd.insert(-1, "-a")
        cmd.insert(-1, str(attachment))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            log_info(f"  [Signal] Message sent to {recipient}")
        else:
            log_error(f"  [Signal] ERROR: {result.stderr.strip()}")
    except FileNotFoundError:
        log_error("  [Signal] ERROR: signal-cli not found. Install from: https://github.com/AsamK/signal-cli")
    except subprocess.TimeoutExpired:
        log_error("  [Signal] ERROR: signal-cli timed out")


# ---------------------------------------------------------------------------
# CLI & main loop
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Augsburg Rollator Detector — monitors webcams for rollators."
    )
    parser.add_argument(
        "--engine", choices=["claude", "yolo"], default="claude",
        help="Detection engine: 'claude' (API, paid) or 'yolo' (local, free). Default: claude"
    )
    parser.add_argument(
        "--interval", type=int, default=CHECK_INTERVAL,
        help=f"Seconds between checks (default: {CHECK_INTERVAL})"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.35,
        help="Min YOLO confidence threshold 0-1 (default: 0.35, only used with --engine yolo)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Open a live window showing both cameras with detection overlays (requires opencv-python)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    interval = args.interval
    engine = args.engine
    show = args.show

    # --- Engine-specific setup ---
    client = None
    yolo_model = None

    if engine == "claude":
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            log_error("ERROR: Set ANTHROPIC_API_KEY environment variable.")
            log_error("Get your key at https://console.anthropic.com")
            return
        client = anthropic.Anthropic(api_key=api_key)
        engine_label = f"Claude ({CLAUDE_MODEL})"
    else:
        yolo_model = load_yolo_model()
        engine_label = f"YOLOv8-nano (local, confidence >= {args.confidence})"

    if show:
        import cv2  # noqa: verify import early
        log_info("  Live preview window enabled (press Q or ESC to quit)")

    # --- Signal setup ---
    signal_sender = os.environ.get("SIGNAL_SENDER")
    signal_recipient = os.environ.get("SIGNAL_RECIPIENT")
    signal_enabled = bool(signal_sender and signal_recipient)

    log_info("=== Augsburg Rollator Detector ===")
    log_info(f"Engine: {engine_label}")
    log_info(f"Live preview: {'ON' if show else 'OFF (use --show to enable)'}")
    log_info(f"Checking {len(CAMERAS)} cameras every {interval}s")
    log_info(f"Signal notifications: {'ON' if signal_enabled else 'OFF (set SIGNAL_SENDER and SIGNAL_RECIPIENT)'}")
    log_info(f"Saving detections to: {SAVE_DIR.resolve()}")
    log_info(f"Log file: {LOG_FILE.resolve()}")
    log_info("Press Ctrl+C to stop.\n")

    quit_requested = False

    try:
        while not quit_requested:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_info(f"--- Check at {now} ---")

            frames = {}  # camera_name -> cv2 image (for --show)

            for name, url in CAMERAS.items():
                log_info(f"  [{name}] Fetching image...")
                image_data = fetch_image(url)
                if image_data is None:
                    continue

                log_info(f"  [{name}] Analyzing ({len(image_data) // 1024} KB) [{engine}]...")
                try:
                    if engine == "claude":
                        result = analyze_image_claude(client, image_data)
                    else:
                        result = analyze_image_yolo(yolo_model, image_data, args.confidence)
                except Exception as e:
                    log_error(f"  [{name}] Analysis error: {e}")
                    continue

                detected = result.get("rollator_detected", False)
                confidence = result.get("confidence", "?")
                desc = result.get("description", "")

                if detected:
                    log_detection(f"  [{name}] >>> ROLLATOR DETECTED ({confidence}) <<<")
                    log_detection(f"  [{name}]     {desc}")

                    saved_path = save_image(name, image_data)
                    log_info(f"  [{name}]     Saved: {saved_path}")

                    if signal_enabled:
                        msg = (
                            f"Rollator detected!\n"
                            f"Camera: {name}\n"
                            f"Time: {now}\n"
                            f"Confidence: {confidence}\n"
                            f"{desc}"
                        )
                        send_signal_message(
                            signal_sender, signal_recipient, msg, saved_path
                        )
                else:
                    log_clear(f"  [{name}] No rollator ({confidence}). {desc}")

                # Build display frame
                if show:
                    if engine == "yolo":
                        frames[name] = draw_yolo_boxes(image_data, result, name)
                    else:
                        frames[name] = draw_claude_overlay(image_data, result, name)

            # Update the live window
            if show and frames:
                quit_requested = show_combined(frames)

            if quit_requested:
                break

            # Wait, but keep the window responsive (check key every 100ms)
            if show:
                import cv2
                waited = 0
                while waited < interval:
                    key = cv2.waitKey(100) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        quit_requested = True
                        break
                    waited += 0.1
            else:
                time.sleep(interval)

            log_info("")

    except KeyboardInterrupt:
        pass

    if show:
        import cv2
        cv2.destroyAllWindows()

    log_info("Stopped.")


if __name__ == "__main__":
    main()
