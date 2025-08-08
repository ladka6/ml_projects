import torch
import torch.nn as nn
import torchvision
import cv2
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader

# Constants
ROOT_DIR = "/traffic_sign/image_data"
CSV_FILE = "traffic_sign/image_data/annotations.csv"
MODEL_SAVE_PATH = "custom_object_detector.pth"


NUM_CLASSES = 43
IMAGE_SIZE = 128


LEARNING_RATE = 0.001
BATCH_SIZE = 4
NUM_EPOCHS = 2

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class CustomObjectDetector(nn.Module):
    def __init__(self, num_classes, img_size=128):
        super(CustomObjectDetector, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flatten_features = 128 * (img_size // 16) * (img_size // 16)

        self.classifier_head = nn.Sequential(
            nn.Linear(self.flatten_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self.bbox_regressor_head = nn.Sequential(
            nn.Linear(self.flatten_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        class_predictions = self.classifier_head(x)
        bbox_predictions = self.bbox_regressor_head(x)
        return class_predictions, bbox_predictions


class TrafficSignDataset(Dataset):
    def __init__(self, csv_file, root_dir, transforms=None):
        self.annotations = pd.read_csv(csv_file, sep=";")
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        record = self.annotations.iloc[idx]
        file_name = record["Filename"]
        subdir, img_dir_name = file_name.split("/")
        img_name = os.path.join(self.root_dir, subdir, img_dir_name)

        image = cv2.imread(img_name)

        if image is None:
            raise ValueError(f"Could not load image: {img_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = record[["Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"]].values.astype("float")

        labels = record["ClassId"]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target


def run_training():
    print(f"Using: {DEVICE}")

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    dataset = TrafficSignDataset(
        csv_file=CSV_FILE, root_dir=ROOT_DIR, transforms=transforms
    )
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CustomObjectDetector(num_classes=NUM_CLASSES, img_size=IMAGE_SIZE).to(
        DEVICE
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    classification_loss_fn = nn.CrossEntropyLoss()
    bbox_loss_fn = nn.MSELoss()
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_epoch_loss = 0

        for i, (images, targets) in enumerate(data_loader):
            images = images.to(DEVICE)
            true_labels = targets["labels"].to(DEVICE)
            true_boxes = targets["boxes"].to(DEVICE)

            pred_labels, pred_boxes = model(images)

            class_loss = classification_loss_fn(pred_labels, true_labels)
            bbox_loss = bbox_loss_fn(pred_boxes, true_boxes)

            loss = class_loss + bbox_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()

        avg_epoch_loss = total_epoch_loss / len(data_loader)
        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_epoch_loss:.4f}, "
            f"Class Loss: {class_loss.item():.4f}, BBox Loss: {bbox_loss.item():.4f}"
        )

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    run_training()
