import torch
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset


datasets.EMNIST.url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"


def _as_tensor_dataset(dataset):
    images = dataset.data.unsqueeze(1).to(dtype=torch.float32).div_(255)
    images = images.transpose(2, 3).contiguous()
    return TensorDataset(images, dataset.targets.to(dtype=torch.long))


def load_data(batch_size, num_workers=4, pin_memory=False):
    train_dataset = datasets.EMNIST(
        root="./data",
        split="balanced",
        train=True,
        download=True,
    )
    train_dataset = _as_tensor_dataset(train_dataset)

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_options["persistent_workers"] = True
        loader_options["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_options,
    )

    dev_dataset = datasets.EMNIST(
        root="./data",
        split="balanced",
        train=False,
        download=True,
    )
    dev_dataset = _as_tensor_dataset(dev_dataset)

    dev_loader = DataLoader(
        dev_dataset,
        shuffle=False,
        **loader_options,
    )

    return train_loader, dev_loader