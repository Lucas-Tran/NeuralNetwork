from torchvision import datasets, transforms
from torch.utils.data import DataLoader


datasets.EMNIST.url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

def load_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda img: img.transpose(1, 2))
    ])

    train_dataset = datasets.EMNIST(
        root="./data",
        split="balanced",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
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
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, dev_loader