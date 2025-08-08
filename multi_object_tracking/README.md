# Multi-Object Tracking (MOT)

## 🎯 Objective
Track multiple moving objects across frames, maintaining unique IDs.

## 📂 Dataset
- **Name:** MOTChallenge or KITTI Tracking
- **Links:** https://motchallenge.net/ or KITTI tracking benchmark
- **Content:** Sequences with bounding boxes and IDs.
- **Download Instructions:**  
Follow dataset instructions.

## 🛠️ Methods
- DeepSORT tracking algorithm
- Detection from YOLOv8 + feature embedding
- MOTA / MOTP metrics

## 🚀 How to Run
```bash
python track.py --weights yolov8s.pt --source data/video.mp4
```

## 📊 Results
*(Add tracking video with ID overlays)*

## 📚 References
- DeepSORT Paper: https://arxiv.org/abs/1703.07402
