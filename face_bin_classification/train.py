import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from torch.nn import BCEWithLogitsLoss
import json


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ModelCNN import ModelCNN
from select_device import select_device
from gradient_decent import gradient_decent
from error_percentage import calculate_error_multiple_classes, calculate_error_binary

transform = transforms.ToTensor()

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

        return image, torch.tensor(label, dtype=torch.long)

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

    model = ModelCNN("architecture.json")
    model.to(device)
    gradient_decent(device, model, train_loader, dev_loader, BCEWithLogitsLoss(), hyperparams_train, error_func=calculate_error_binary)
    torch.save(model.state_dict(), "model.pth")