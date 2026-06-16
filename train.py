import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_path = './data/asl_alphabet_train/asl_alphabet_train'
dataset = datasets.ImageFolder(data_path, transform=data_transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# downloading the model MobileNetV2
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# freezing base layers
for param in model.parameters():
    param.requires_grad = False

# replacing the classifier head for our 29 ASL classes
num_features = model.classifier[1].in_features
num_classes = len(dataset.classes)
model.classifier[1] = nn.Linear(num_features, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)

epochs = 3

print("Starting training...")

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1} / {epochs} | Loss: {running_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), 'mobilenet_v2_asl.pth')

print(f"Training finished, model saved to mobilenet_v2_asl.pth")