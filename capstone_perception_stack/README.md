# Capstone: Autonomous Driving Perception Stack

## 🎯 Objective
Integrate all perception modules into a unified ROS2 pipeline using CARLA simulator.

## 📂 Dataset
- **Name:** CARLA Simulation Data
- **Link:** https://carla.org/
- **Content:** Simulated RGB, LiDAR, depth data with annotations.
- **Download Instructions:**  
Generate data from CARLA with custom scenarios.

## 🛠️ Modules
- Detection: YOLOv8
- Segmentation: DeepLabV3+
- Tracking: DeepSORT
- Fusion: LiDAR + Camera

## 🚀 How to Run
```bash
ros2 launch perception_stack.launch.py
```

## 📊 Results
*(Include full pipeline output video)*

## 📚 References
- CARLA Docs: https://carla.readthedocs.io/
- ROS2 Tutorials: https://docs.ros.org/en/foxy/Tutorials.html
