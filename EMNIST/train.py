from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
from torch.nn import CrossEntropyLoss
import torch
import kornia.augmentation as K


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ModelCNN import ModelCNN
from select_device import select_device
from gradient_decent import gradient_decent
from error_percentage import calculate_error_multiple_classes

def transpose_emnist(x):
  return x.transpose(1, 2)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(transpose_emnist),
])

datasets.EMNIST.url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

if __name__ == "__main__":
    with open("hyperparameters.json", "r") as file:
        hyperparameters = json.load(file)
    hyperparameters_data = hyperparameters["data"]
    hyperparams_train = hyperparameters["train"]
    hyperparams_augmentation = hyperparameters["augmentation"]

    train_dataset = datasets.EMNIST(
        root=".",
        train=True,
        split="balanced",
        transform=transform,
        download=True,
    )
    dev_dataset = datasets.EMNIST(
        root=".",
        train=False,
        split="balanced",
        transform=transform,
        download=True,
    )

    device = select_device()
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=hyperparameters_data["batch_size"],
        num_workers=hyperparameters_data["num_workers"],
        persistent_workers=hyperparameters_data["num_workers"] > 0,
        pin_memory=pin_memory,
    )

    dev_loader = DataLoader(
        dataset=dev_dataset,
        shuffle=False,
        batch_size=hyperparameters_data["batch_size"],
        num_workers=hyperparameters_data["num_workers"],
        persistent_workers=hyperparameters_data["num_workers"] > 0,
        pin_memory=pin_memory,
    )

    augmentation=K.AugmentationSequential(
        K.RandomRotation(
            degrees=hyperparams_augmentation["rotate"]["by"],
            p=hyperparams_augmentation["rotate"]["prob"]
        ),
        K.RandomAffine(
            degrees=0,
            translate=(
                hyperparams_augmentation["translate"]["by"][0],
                hyperparams_augmentation["translate"]["by"][1]
            ),
            p=hyperparams_augmentation["translate"]["prob"],
        ),
        K.RandomAffine(
            degrees=0,
            scale=(
                hyperparams_augmentation["scale"]["by"][0],
                hyperparams_augmentation["scale"]["by"][1]
            ),
            p=hyperparams_augmentation["scale"]["prob"],
        )
    ).to(device)

    model = ModelCNN("architecture.json")
    model.to(device)
    gradient_decent(
        device,
        model,
        train_loader,
        dev_loader,
        CrossEntropyLoss(),
        hyperparams_train,
        error_func=calculate_error_multiple_classes,
        augmentation=augmentation
    )
    torch.save(model.state_dict(), "model.pth")