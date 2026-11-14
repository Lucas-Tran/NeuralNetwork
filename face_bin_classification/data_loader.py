import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset

transform = transforms.ToTensor()

class FaceDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        image = item["image"].convert("RGB")
        label = int(item["label"])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32).unsqueeze(0)

def load_data(batch_size, num_workers, pin_memory):
    dataset = load_dataset("theoriclabs/face-or-not")
    train_dataset = FaceDataset(dataset["train"], transform)
    dev_dataset = FaceDataset(dataset["validation"], transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )

    return train_loader, dev_loader