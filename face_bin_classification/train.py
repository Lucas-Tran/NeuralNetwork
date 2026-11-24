import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from torch.nn import BCEWithLogitsLoss
import kornia.augmentation as K
import json


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ModelCNN import ModelCNN
from select_device import select_device
from gradient_decent import gradient_decent
from error_percentage import calculate_error_binary

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


class FaceDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]

        image = item["image"].convert("RGB")
        label = item["label"]

        image = transform(image)

        return image, torch.tensor(label, dtype=torch.float32).unsqueeze(0)

if __name__ == "__main__":
    with open("hyperparameters.json", "r") as f:
        hyperparams = json.load(f)
    hyperparams_train = hyperparams["train"]
    hyperparams_data = hyperparams["data"]

    dataset = load_dataset("theoriclabs/face-or-not")
    train_dataset = FaceDataset(dataset["train"])
    dev_dataset = FaceDataset(dataset["validation"])

    device = select_device()
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparams_data["batch_size"],
        shuffle=True,
        num_workers=hyperparams_data["num_workers"],
        pin_memory=pin_memory,
        persistent_workers=hyperparams_data["num_workers"] > 0
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=hyperparams_data["batch_size"],
        shuffle=False,
        num_workers=hyperparams_data["num_workers"],
        pin_memory=pin_memory,
        persistent_workers=hyperparams_data["num_workers"] > 0
    )

    train_augmentation = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomRotation(degrees=15, p=0.5),
        K.RandomResizedCrop(
            size=(128, 128),
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
            p=0.5
        ),
        K.ColorJiggle(
            brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.03,
            p=0.5
        ),
        K.RandomGrayscale(p=0.15),
        K.RandomErasing(
            scale=(0.02, 0.08),
            ratio=(0.5, 2.0),
            value=0.0,
            p=0.15
        ),

        data_keys=["input"]
    ).to(device)

    model = ModelCNN("architecture.json")
    model.to(device)
    gradient_decent(
        device,
        model,
        train_loader,
        dev_loader,
        BCEWithLogitsLoss(),
        hyperparams_train,
        error_func=calculate_error_binary,
        augmentation=train_augmentation
    )
    torch.save(model.state_dict(), "model.pth")