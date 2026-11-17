from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
from torch.nn import CrossEntropyLoss
import torch


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ModelCNN import ModelCNN
from select_device import select_device
from gradient_decent import gradient_decent
from error_percentage import calculate_error_multiple_classes

if __name__ == "__main__":
    with open("hyperparameters.json", "r") as file:
        hyperparameters = json.load(file)
    hyperparameters_data = hyperparameters["data"]
    hyperparams_train = hyperparameters["train"]

    train_dataset = datasets.MNIST(
        root=".",
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )
    dev_dataset = datasets.MNIST(
        root=".",
        train=False,
        transform=transforms.ToTensor(),
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

    model = ModelCNN("architecture.json")
    model.to(device)
    gradient_decent(device, model, train_loader, dev_loader, CrossEntropyLoss(), hyperparams_train, calculate_error_multiple_classes)
    torch.save(model.state_dict(), "model.pth")