# Object Detection with YOLOv8 / Faster R-CNN

## 🎯 Objective
Detect vehicles, pedestrians, and traffic signs in driving scenes.  
This is a core perception task in autonomous driving.

## 📂 Dataset
- **Name:** KITTI Object Detection
- **Link:** http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
- **Content:** RGB images with 2D bounding box annotations.
- **Download Instructions:**  
Follow KITTI website instructions.

## 🛠️ Methods
- YOLOv8 training pipeline
- Alternative: Faster R-CNN with PyTorch
- mAP evaluation

## 🚀 How to Run
```bash
yolo detect train data=kitti.yaml model=yolov8s.pt epochs=50
```

## 📊 Results
*(Add precision-recall curves, detection visualizations)*

## 📚 References
- YOLOv8 Docs: https://docs.ultralytics.com/
- Faster R-CNN Tutorial: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
