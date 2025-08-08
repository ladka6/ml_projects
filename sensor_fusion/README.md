# LiDAR + Camera Sensor Fusion

## 🎯 Objective
Combine LiDAR point clouds and camera images for improved detection.

## 📂 Dataset
- **Name:** KITTI 3D + Calibration
- **Link:** http://www.cvlibs.net/datasets/kitti/raw_data.php
- **Content:** Synchronized LiDAR + RGB images.
- **Download Instructions:**  
Follow KITTI calibration download.

## 🛠️ Methods
- Extrinsic & intrinsic calibration
- Projecting LiDAR points to image plane
- Late fusion for detection

## 🚀 How to Run
```bash
python fusion_demo.py --lidar data/000000.bin --image data/000000.png
```

## 📊 Results
*(Overlay LiDAR points on RGB images)*

## 📚 References
- Sensor Fusion Guide: https://towardsdatascience.com/lidar-camera-fusion-in-the-wild-a-practical-guide-2019-7c9f855d9c8f
