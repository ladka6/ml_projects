# 3D Object Detection from LiDAR

## 🎯 Objective
Detect 3D bounding boxes around vehicles and pedestrians from LiDAR data.

## 📂 Dataset
- **Name:** nuScenes
- **Link:** https://www.nuscenes.org/
- **Content:** 3D point clouds + images + annotations.
- **Download Instructions:**  
Register and download nuScenes mini/full dataset.

## 🛠️ Methods
- PointPillars or CenterPoint architectures
- 3D IoU metric

## 🚀 How to Run
```bash
python train_pointpillars.py --dataset data/nuscenes
```

## 📊 Results
*(Add 3D detection visualizations)*

## 📚 References
- PointPillars Paper: https://arxiv.org/abs/1812.05784
