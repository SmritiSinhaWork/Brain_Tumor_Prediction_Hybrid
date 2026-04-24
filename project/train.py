import kagglehub
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from dmd import dmd_process
import cv2
import numpy as np
from model import HybridModel
import torchvision.transforms as transforms
from torch.utils.data import random_split

path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

train_dir = os.path.join(path, "Training")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

class DMDImageDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        path, label = self.samples[index]
        
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (64, 64))
        
        img = dmd_process(img)
        img = np.nan_to_num(img)
        img = (img - img.mean()) / (img.std() + 1e-8)

        img = transform(img)
        img = (img - img.mean()) / (img.std() + 1e-8)

        return img, label

train_data = DMDImageDataset(train_dir)

train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size

train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HybridModel()
model = model.to(device)

targets = [label for _, label in train_dataset]
class_counts = np.bincount(targets)

weights = (1.0 / torch.tensor(class_counts, dtype=torch.float32)).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.00005)

model.train()

best_acc = 0

for epoch in range(40):
    correct = 0
    total = 0
    epoch_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total

    accuracy = 100 * correct / total
    avg_loss = epoch_loss / len(train_loader)

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")

    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(model.state_dict(), "model.pth")
    
model.train()