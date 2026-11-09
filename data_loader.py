from torchvision import datasets, transforms
from torch.utils.data import DataLoader


datasets.EMNIST.url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"


def _transpose_image(image):
    return image.transpose(1, 2)


def load_data(batch_size, num_workers=4, pin_memory=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(_transpose_image)
    ])

    train_dataset = datasets.EMNIST(
        root="./data",
        split="balanced",
        train=True,
        download=True,
        transform=transform
    )

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
        transform=transform
    )

    dev_loader = DataLoader(
        dev_dataset,
        shuffle=False,
        **loader_options,
    )

    return train_loader, dev_loader