# YOLOv8 Video Tracker

This is a professional project developed on demand, focused on **object detection and tracking in video** using **YOLOv8** and **Deep SORT**.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/notfarchi/track_counts
cd track_count
```

### 2. (Optional) Create and activate a virtual environment

```bash
# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install ultralytics opencv-python deep_sort_realtime
```

## How to Use

1. Add the input video

   Place your video in the project root directory and name it video.mp4.

   Important: the input file must be named video.mp4. Other names or formats will not be recognized by the script.

2. Run the main script

```bash
python track.py
```

3. Check the result

The result will be automatically saved in the file:

```bash
resultado_contagem.txt
```

Attention: Always rename the input video to video.mp4 before running the script.
