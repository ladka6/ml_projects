# Video Frame Extraction & Annotation

## 🎯 Objective
Prepare annotated datasets for detection/tracking tasks.  
Learn how to extract frames from video and label objects using annotation tools.

## 📂 Dataset
- **Source:** BDD100K Dashcam Videos
- **Link:** https://bdd-data.berkeley.edu/
- **Content:** Driving videos with diverse weather & lighting.
- **Download Instructions:**  
Filter a small subset for annotation.

## 🛠️ Methods
- Frame extraction using OpenCV
- Manual annotation with LabelImg or CVAT
- YOLO/COCO annotation format

## 🚀 How to Run
```bash
python extract_frames.py --video input.mp4 --output frames/
```

## 📊 Results
*(Include screenshot of annotated frames)*

## 📚 References
- LabelImg: https://github.com/heartexlabs/labelImg
- CVAT: https://cvat.org/
