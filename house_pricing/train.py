import torch
from torch.utils.data import Dataset
import json
from torch.nn import MSELoss

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ModelNN import ModelNN
from gradient_decent import gradient_decent
from select_device import select_device

class HousePriceDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        with open(data_path, "r") as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        x = torch.tensor(
            [example["house_size"],
                  example["num_bedrooms"],
                  example["num_bathrooms"],
                  example["house_age"]],
                 dtype=torch.float32)
        y = torch.tensor([example["estimated_price"] / 1000000], dtype=torch.float32)
        return x, y

if __name__ == "__main__":
    with open("hyperparameters.json", "r") as file:
        hyperparameters = json.load(file)
    hyperparams_data = hyperparameters["data"]
    hyperparams_train = hyperparameters["train"]

    train_dataset = HousePriceDataset("data/train.json")
    dev_dataset = HousePriceDataset("data/dev.json")

    device = select_device()
    pin_memory = device.type == "cuda"

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=hyperparams_data["batch_size"],
        num_workers=hyperparams_data["num_workers"],
        persistent_workers=hyperparams_data["num_workers"] > 0,
        pin_memory=pin_memory,
    )

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        shuffle=False,
        batch_size=hyperparams_data["batch_size"],
        num_workers=hyperparams_data["num_workers"],
        persistent_workers=hyperparams_data["num_workers"] > 0,
        pin_memory=pin_memory,
    )


    model = ModelNN("architecture.json")
    model.to(device)
    gradient_decent(device, model, train_loader, dev_loader, MSELoss(), hyperparams_train)
    torch.save(model.state_dict(), "model.pth")