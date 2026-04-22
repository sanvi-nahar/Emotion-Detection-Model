import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np

DATA_DIR = os.path.expanduser('~/Downloads/archive/Gesture Image Data')

class ASLDataset(Dataset):
    def __init__(self, data_dir, transform=None, augment=False):
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d != '_'])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_path = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_path)[:100]:
                self.samples.append((os.path.join(class_path, img_name), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.augment and random.random() > 0.5:
            brightness = random.uniform(0.7, 1.3)
            image = Image.fromarray(np.clip(np.array(image) * brightness, 0, 255).astype(np.uint8))
            
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = image.rotate(angle, fillcolor=(255, 255, 255))
            
            if random.random() > 0.5:
                image = image.resize((int(32 * random.uniform(0.9, 1.1)), int(32 * random.uniform(0.9, 1.1))))
            
            shift_x = random.randint(-3, 3)
            shift_y = random.randint(-3, 3)
            from PIL import ImageOps
            image = ImageOps.expand(image, border=(shift_x, shift_y), fill=(255, 255, 255))
        
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

train_dataset = ASLDataset(DATA_DIR, transform=transform, augment=True)
test_dataset = ASLDataset(DATA_DIR, transform=transform, augment=False)

print(f"Total: {len(train_dataset)}, Classes: {train_dataset.classes}")

indices = list(range(len(train_dataset)))
random.shuffle(indices)
train_indices = indices[:int(0.8 * len(indices))]
test_indices = indices[int(0.8 * len(indices)):]

train_set = torch.utils.data.Subset(train_dataset, train_indices)
test_set = torch.utils.data.Subset(test_dataset, test_indices)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

class ASLModel(nn.Module):
    def __init__(self, num_classes):
        super(ASLModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = ASLModel(len(train_dataset.classes))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nTraining with augmentation...")
for epoch in range(8):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_acc = 100. * correct / total
    print(f"Epoch {epoch+1}/15 - Loss: {running_loss/len(train_loader):.4f} - Acc: {train_acc:.2f}%")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_acc = 100. * correct / total
print(f"\nTest Accuracy: {test_acc:.2f}%")

torch.save({
    'model': model.state_dict(),
    'classes': train_dataset.classes
}, 'sign_model.pth')
print("Model saved to sign_model.pth")