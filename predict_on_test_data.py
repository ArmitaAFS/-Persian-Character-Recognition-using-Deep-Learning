import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, datasets


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Classes
dataset = datasets.ImageFolder("dataset/train_data")
data_classes = dataset.classes

# Load Labels JSON
with open("labels.json", "r", encoding="utf-8") as f:
    labels = json.load(f)


# Model
class DeepModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



# Load Model
model = DeepModel(len(data_classes)).to(device)

model.load_state_dict(
    torch.load(
        "Alpha_Recognation_Model.pth",
        map_location=device
    )
)

model.eval()


# Image Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load Image
image_path = "dataset/test_data/16/BKoodakO.png" 

image = Image.open(image_path).convert("RGB")

image = transform(image)
image = image.unsqueeze(0)
image = image.to(device)


# Predict
with torch.no_grad():

    outputs = model(image)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted_idx = torch.max(
        probabilities,
        dim=1
    )

    predicted_idx = predicted_idx.item()
    confidence = confidence.item() * 100

    class_id = data_classes[predicted_idx]

    character = labels[class_id]


# Result
print("Device:", device)
print("Index:", predicted_idx)
print("Class ID:", class_id)
print("Character:", character)
print(f"Confidence: {confidence:.2f}%")