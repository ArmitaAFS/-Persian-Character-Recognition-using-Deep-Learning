import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Set Hyper-parameters
batch_size = 64
image_size = (128, 128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),  # scales to [0,1]
])

full_dataset = datasets.ImageFolder(
    root="dataset/train_data",
    transform=transform
)

train_size = int(0.8 * len(full_dataset))
valid_size = len(full_dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, valid_dataset = random_split(
    full_dataset,
    [train_size, valid_size],
    generator=generator
)

test_dataset = datasets.ImageFolder(
    root="dataset/test_data",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

data_classes = full_dataset.classes
print(data_classes)

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


deep_model = DeepModel(len(data_classes)).to(device)

print(deep_model)

# Loss / Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(deep_model.parameters())


# Early Stopping
patience = 10
min_delta = 0.001

best_val_loss = float("inf")
best_model_weights = None
patience_counter = 0

history = {
    "accuracy": [],
    "val_accuracy": [],
    "loss": [],
    "val_loss": []
}

epochs = 100


# Training Loop
for epoch in range(epochs):

    #Train
    deep_model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = deep_model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total

    #Validation
    deep_model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = deep_model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    history["accuracy"].append(train_acc)
    history["val_accuracy"].append(val_acc)
    history["loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"loss={train_loss:.4f} "
        f"acc={train_acc:.4f} "
        f"val_loss={val_loss:.4f} "
        f"val_acc={val_acc:.4f}"
    )

    #Model Checkpoint
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        best_model_weights = copy.deepcopy(deep_model.state_dict())

        torch.save(
            best_model_weights,
            "Alpha_Recognation_Model.pth"
        )

        print("Model improved -> saved")

        patience_counter = 0
    else:
        patience_counter += 1

    #Early Stopping
    if patience_counter >= patience:
        print("Early stopping triggered")
        break

# Restore best weights
if best_model_weights is not None:
    deep_model.load_state_dict(best_model_weights)

# Test
deep_model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = deep_model(images)
        preds = outputs.argmax(dim=1)

        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

accuracy = test_correct / test_total

print(f"test accuracy: {accuracy:.4f}")



# Plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history["accuracy"], label="Train Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()