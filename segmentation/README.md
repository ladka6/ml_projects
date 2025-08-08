# Semantic Segmentation of Drivable Areas

## 🎯 Objective
Segment drivable road areas from images.  
This is used by autonomous vehicles to understand road boundaries and obstacles.

## 📂 Dataset
- **Name:** Cityscapes
- **Link:** https://www.cityscapes-dataset.com/
- **Content:** High-resolution street scenes with pixel-level labels.
- **Download Instructions:**  
Register on website and download dataset.

## 🛠️ Methods
- U-Net or DeepLabV3+ architecture
- Cross-entropy loss
- IoU evaluation metric

## 🚀 How to Run
```bash
python train_deeplab.py --dataset data/cityscapes --epochs 50
```

## 📊 Results
*(Add IoU scores, segmentation overlays)*

## 📚 References
- DeepLabV3+: https://arxiv.org/abs/1802.02611
- U-Net: https://arxiv.org/abs/1505.04597
