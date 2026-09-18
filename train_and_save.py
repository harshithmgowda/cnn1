"""
Train the CIFAR-10 CNN and save weights to models/cifar10_cnn.pth
Matches the exact architecture used in app.py and sample.ipynb.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# ==========================================
# 1. IMAGE PREPROCESSING
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# ==========================================
# 2. DATASET
# ==========================================

trainset = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

testset = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# ==========================================
# 3. DATALOADERS
# ==========================================

trainloader = DataLoader(
    trainset,
    batch_size=64,
    shuffle=True
)

testloader = DataLoader(
    testset,
    batch_size=64,
    shuffle=False
)


# ==========================================
# 4. CNN MODEL (same as app.py)
# ==========================================

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(4 * 4 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# ==========================================
# 5. SETUP
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ==========================================
# 6. TRAINING
# ==========================================

epochs = 10

for epoch in range(epochs):
    model.train()
    epoch_training_loss = 0.0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_training_loss += loss.item()

    average_loss = epoch_training_loss / len(trainloader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")


# ==========================================
# 7. EVALUATION
# ==========================================

correct_labels = 0
total_labels = 0

model.eval()

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct_labels += (predicted == labels).sum().item()
        total_labels += labels.size(0)

accuracy = correct_labels / total_labels
print(f"Accuracy = {accuracy * 100:.2f}%")


# ==========================================
# 8. SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/cifar10_cnn.pth")
print("Model saved to models/cifar10_cnn.pth")
