# Vehicle Classification with Transfer Learning

## 🎯 Objective
Classify vehicle make & model using transfer learning from pretrained CNNs (ResNet, EfficientNet).  
This skill is important for traffic analysis and autonomous fleet monitoring.

## 📂 Dataset
- **Name:** Stanford Cars Dataset
- **Link:** https://ai.stanford.edu/~jkrause/cars/car_dataset.html
- **Content:** 196 classes of cars, ~16,000 images.
- **Download Instructions:**  
Register and download from the link above.

## 🛠️ Methods
- Transfer learning from ResNet50
- Fine-tuning last layers
- Data augmentation for better generalization

## 🚀 How to Run
```bash
python train.py --dataset data/stanford_cars --epochs 20 --model resnet50
```

## 📊 Results
*(Add accuracy table, confusion matrix, sample predictions)*

## 📚 References
- PyTorch Transfer Learning Tutorial: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
