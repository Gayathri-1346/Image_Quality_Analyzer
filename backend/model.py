import os
import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier

class ImageAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ImageDataset(Dataset):
    def __init__(self, folder):
        self.files = []
        for root, folders, files in os.walk(folder):
            for filename in files:
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    self.files.append(
                        os.path.join(root, filename)
                    )
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()])
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        path = self.files[index]
        image = cv2.imread(path)
        if image is None:
            raise ValueError("Could not read image")
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
        image = self.transform(image)
        return image


def train_autoencoder( train_folder="../data/seg_train", epochs=10,batch_size=16):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dataset = ImageDataset(train_folder)
    if len(dataset) == 0:
        raise ValueError(
            "No images found in training folder"
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    model = ImageAutoencoder().to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images in loader:
            images = images.to(device)
            reconstructed = model(images)
            loss = loss_function(
                reconstructed,
                images
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(loader)
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {average_loss:.4f}"
        )
    os.makedirs("../models", exist_ok=True)
    torch.save(
        model.state_dict(),
        "../models/autoencoder.pth"
    )
    print("Autoencoder saved.")

def load_autoencoder(
    model_path="../models/autoencoder.pth"
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = ImageAutoencoder().to(device)
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )
    model.eval()
    return model, device

def reconstruction_error(
    model,
    image,
    device
):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )
    tensor = transform(rgb_image)
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        reconstructed = model(tensor)
        error = torch.mean(
            (tensor - reconstructed) ** 2
        )
    return float(error.item())

FEATURE_NAMES = [
    "sharpness",
    "exposure",
    "brightness",
    "underexposed_ratio",
    "overexposed_ratio",
    "noise",
    "noise_level",
    "contrast",
    "contrast_value",
    "saturation",
    "mean_saturation",
    "reconstruction_error"
]

def train_fusion_model(X, y):
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X, y)
    os.makedirs("../models", exist_ok=True)
    joblib.dump(
        model,
        "../models/quality_classifier.pkl"
    )
    print("Random Forest saved.")
    return model


def load_fusion_model(
    model_path="../models/quality_classifier.pkl"
):
    return joblib.load(model_path)

def predict_quality(
    model,
    feature_values
):
    values = np.array(
        [feature_values[name] for name in FEATURE_NAMES]
    ).reshape(1, -1)
    prediction = model.predict(values)[0]
    probabilities = model.predict_proba(values)[0]
    probability_map = dict(
        zip(model.classes_, probabilities)
    )
    return prediction, probability_map