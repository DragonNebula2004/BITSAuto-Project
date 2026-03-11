# 🚗 BITSAuto — Road Segmentation & Object Detection

A computer vision pipeline for autonomous driving. The system uses a custom-trained **YOLOv8 segmentation model** to detect and segment roads in real time, combined with a pretrained YOLOv8 detection model to identify obstacles on the road. Designed to run both on standard hardware and on **Raspberry Pi**.

---

## 📌 Features

- **Road Segmentation** — Fine-tuned YOLOv8n-seg model trained on BITS Goa campus roads
- **Object Detection** — Pretrained YOLOv8n (COCO) detects obstacles, pedestrians, and vehicles on the road
- **Lane Midpoint Estimation** — Draws a center divider on the detected road bounding box to estimate lane center
- **On-Road Filtering** — Object detection results are filtered to only flag objects within the segmented road region
- **Raspberry Pi Support** — Dedicated script optimized for live camera input with frame-skipping for performance
- **Roboflow Integration** — Dataset annotated using Roboflow for streamlined custom training

---

## 🗂️ Project Structure

```
BITSAuto-Project/
│
├── train_model.py              # Fine-tune YOLOv8 on the custom BITS Goa dataset
├── segment_road.py             # Road segmentation + bounding box + midpoint on video
├── segment_detect.py           # Road segmentation + on-road object detection on video
├── segment_detect_raspi.py     # Live camera version optimized for Raspberry Pi
├── trained_model.pt            # (Downloaded separately) Fine-tuned segmentation model
└── datasets/                   # (Downloaded separately) Custom annotated dataset
```

---

## ⚙️ Requirements

Install dependencies with:

```bash
pip install ultralytics opencv-python numpy
```

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 model training and inference |
| `opencv-python` | Video capture, frame processing, and display |
| `numpy` | Mask array manipulation |

> Python 3.8+ recommended.

---

## 📥 Downloads

The model weights and dataset are hosted on Google Drive:

🔗 **[Download Dataset & Model](https://drive.google.com/drive/folders/1_MA48VKG8hAU8YRUScmKIu83-DeEHZZM?usp=drive_link)**

| File | Description |
|---|---|
| `datasets/` | Custom annotated road dataset (BITS Goa) with `data.yaml` |
| `trained_model.pt` | Fine-tuned YOLOv8 segmentation model |
| `IMG_3010.mp4` | Sample test video for running inference |

Place downloaded files in the project root before running any scripts.

---

## 🚀 Usage

### 1. Train the Model

Fine-tunes a pretrained `yolov8n-seg.pt` on the custom BITS Goa dataset for 5 epochs and saves the result as `trained_model.pt`.

```bash
python train_model.py
```

> Requires the `datasets/` folder with a valid `data.yaml` configuration file.

---

### 2. Road Segmentation on Video

Runs road segmentation and draws the road bounding box with a center lane divider on the test video.

```bash
python segment_road.py
```

> Requires `trained_model.pt` and `IMG_3010.mp4` in the project root.

---

### 3. Segmentation + Object Detection on Video

Combines road segmentation with YOLOv8 object detection. Only objects detected **within the road region** are flagged.

```bash
python segment_detect.py
```

> Requires `trained_model.pt`, `yolov8n.pt` (auto-downloaded by ultralytics), and `IMG_3010.mp4`.

---

### 4. Live Camera (Raspberry Pi)

Runs the full segmentation + detection pipeline on a **live camera feed**, with frame-skipping tuned for Raspberry Pi performance.

```bash
python segment_detect_raspi.py
```

> Uses camera index `0`. Press `q` to quit.  
> Adjust `skip_frame` in the script to tune performance vs. latency on your hardware.

---

## 🧠 How It Works

```
Camera / Video Input
        │
        ▼
┌──────────────────────┐
│  Road Segmentation   │  ← Custom YOLOv8n-seg (trained_model.pt)
│  - Pixel-level mask  │
│  - Bounding box      │
│  - Lane midpoint     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Object Detection    │  ← Pretrained YOLOv8n (COCO)
│  - Filter to road    │
│  - Draw boxes        │
└──────────┬───────────┘
           │
           ▼
    Annotated Frame Output
```

The segmentation mask is overlaid on each frame at 30% opacity. The largest detected road region is used as a reference bounding box, and any object whose center falls within that box is flagged as an on-road obstacle.

---

## 📚 References & Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8/)
- [YOLO Tutorial Playlist](https://www.youtube.com/playlist?list=PL1u-h-YIOL0sZJsku-vq7cUGbqDEeDK0a)
- [Roboflow — Dataset Annotation](https://roboflow.com/)

---

## 🏫 About

This project was developed as part of the **BITSAuto** initiative at **BITS Pilani, Goa Campus**, working toward autonomous vehicle research on real campus road conditions.




