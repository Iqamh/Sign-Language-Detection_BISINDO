# BISINDO Sign Language Detection with YOLOv8

This project focuses on detecting **Indonesian Sign Language (BISINDO)** gestures using computer vision and deep learning. It utilizes **YOLOv8** for object detection to recognize both **letters** and **words** from live camera input in real time.

## 🧰 Tools and Libraries Used

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- `OpenCV` (`cv2`) – for video frame capture and webcam access
- `Matplotlib` – for visualization
- `scikit-learn` – for classification metrics
- `PyYAML`, `os`, `shutil`, `random`, `numpy` – for utility tasks

## Dataset Details

- The dataset is sourced from **Kaggle**

- It includes two parts:
  - **Alphabets/**: Images of BISINDO letters A–Z.
  - **Words/**: Videos representing BISINDO words (e.g., "Aku", "Dia", "Mana"), which are converted to image frames.

## How It Works

1. **Data Preparation**:
   - `split_data.py` splits the dataset by folder, keeping all frames from one video together.
   - All labels follow YOLO format (bounding boxes with class IDs).
