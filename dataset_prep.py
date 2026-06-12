import torch
from torchvision import datasets, transforms
import json
import os

data_dir = './data/asl_alphabet_train/asl_alphabet_train'

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if not os.path.exists(data_dir):
    print(f"Ścieżką {data_dir} nie istnieje.")
else:
    train_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    with open('mapping.json', 'w') as f:
        json.dump(idx_to_class, f, indent=4)

    print("Utworzono plik mapping.json.")