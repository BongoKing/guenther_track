# Günther Track — Augsburg Rollator Detector

Monitors live webcam streams from Augsburg's city center and detects people using rollators (wheeled walkers). When a rollator is spotted, the image is saved and an optional Signal notification is sent. The half-ready tool was vibe-coded with Claude 4.5 rather fast, as we were searching for a missing person with a rollator. I will not be further maintained by BongoKing, but I thought it was share-worthy as someone might get inspired by it.

## Webcams

|Camera|Location|
|-|-|
|Rathausplatz (Dachspitz)|Overlooking the town hall square|
|Rathaus \& Perlachturm|View of the town hall and Perlach tower|

## Detection Engines

Choose your engine via the `--engine` flag:

|Engine|Cost|Requirements|Accuracy|
|-|-|-|-|
|`claude`|Paid (Anthropic API)|`ANTHROPIC\\\_API\\\_KEY`|High — Claude understands "rollator" directly|
|`yolo`|Free (runs locally)|CPU or GPU|Medium — heuristic based on COCO object classes|

## Installation

```bash
# Clone the repo
git clone https://github.com/YOUR\\\_USERNAME/guenther\\\_track.git
cd guenther\\\_track

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\\\\Scripts\\\\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### Engine-specific installs

```bash
# Claude mode only
pip install anthropic

# YOLO mode only (downloads \\\~6 MB model on first run)
pip install ultralytics Pillow

# Live preview window (--show)
pip install opencv-python
```

## Usage

### Claude engine (default)

```bash
# Set your API key
export ANTHROPIC\\\_API\\\_KEY="sk-ant-..."          # Linux/macOS
$env:ANTHROPIC\\\_API\\\_KEY = "sk-ant-..."          # PowerShell

python rollator\\\_detector.py --engine claude
```

### YOLO engine (free, offline)

```bash
python rollator\\\_detector.py --engine yolo
```

### Live preview window

Add `--show` to open a live window displaying both camera feeds side-by-side with detection overlays:

```bash
# YOLO + live window (bounding boxes around every detected object)
python rollator\\\_detector.py --engine yolo --show

# Claude + live window (status overlay — no bounding boxes since Claude doesn't return coordinates)
python rollator\\\_detector.py --engine claude --show
```

**What you'll see:**

* Both cameras displayed **side-by-side** in a single window
* **YOLO mode**: colored bounding boxes around every detected object

  * 🔴 **Red** = rollator match (person + walker-like object nearby)
  * 🟡 **Orange** = rollator hint object (suitcase, chair, bench, etc.)
  * 🔵 **Cyan** = person
  * ⚪ **Grey** = other objects
* **Claude mode**: status banner + description overlay (red border if detected)
* Press **Q** or **ESC** to quit

### All options

```
python rollator\\\_detector.py --help

options:
  --engine {claude,yolo}   Detection engine (default: claude)
  --interval SECONDS       Seconds between checks (default: 30)
  --confidence FLOAT       YOLO min confidence 0-1 (default: 0.35)
  --show                   Open live preview window with detection overlays
```

## Signal Notifications (optional, not yet tested)

Requires [signal-cli](https://github.com/AsamK/signal-cli) to be installed and registered.

```bash
export SIGNAL\\\_SENDER="+491234567890"
export SIGNAL\\\_RECIPIENT="+490987654321"
```

When a rollator is detected, you'll receive a Signal message with the camera name, timestamp, confidence, and the captured image as an attachment.

## Output

* **Console**: Detections are shown in **red**, clear results in **green**
* **Log file**: `rollator\\\_detector.log` — full timestamped log of every check
* **Saved images**: `rollator\\\_detections/` — JPEGs saved when a rollator is detected
* **Live window** (`--show`): Both cameras side-by-side with bounding boxes / overlays

## How YOLO detection works

YOLOv8 (COCO dataset) has no dedicated "rollator" class. The detector uses a heuristic: it looks for a **person** detected near objects that YOLO commonly confuses with rollators (suitcase, chair, bench, etc.). When a person and such an object are in close proximity, it flags a possible rollator sighting. Adjust sensitivity with `--confidence`.
License
---

MIT License — see [LICENSE](LICENSE).

