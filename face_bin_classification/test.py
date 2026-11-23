from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from datasets import load_dataset
import json
import math
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from select_device import select_device
from ModelCNN import ModelCNN

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

        return image, torch.tensor(label, dtype=torch.float32).unsqueeze(0)

if __name__ == "__main__":
    with open("hyperparameters.json", "r") as f:
        hyperparams_data = json.load(f)["data"]
    dataset = load_dataset("theoriclabs/face-or-not")
    test_dataset = FaceDataset(dataset["test"])

    device = select_device()

    pin_memory = device.type == "cuda"

    test_loader = DataLoader(
        test_dataset,
        batch_size=hyperparams_data["batch_size"],
        shuffle=False,
        num_workers=hyperparams_data["num_workers"],
        pin_memory=pin_memory,
        persistent_workers=hyperparams_data["num_workers"] > 0
    )

    model = ModelCNN("architecture.json")
    model.to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
    model.eval()

    TP = torch.zeros((), device=device, dtype=torch.int64)
    FP = torch.zeros((), device=device, dtype=torch.int64)
    TN = torch.zeros((), device=device, dtype=torch.int64)
    FN = torch.zeros((), device=device, dtype=torch.int64)

    misclassified = []

    with torch.inference_mode():
        for X, Y in test_loader:
            X = X.to(device, non_blocking=device.type == "cuda")
            Y = Y.to(device, non_blocking=device.type == "cuda")

            logits = model(X)

            predicted_positive = logits > 0
            actual_positive = Y > 0.5

            TP += (predicted_positive & actual_positive).sum()
            FP += (predicted_positive & ~actual_positive).sum()
            TN += (~predicted_positive & ~actual_positive).sum()
            FN += (~predicted_positive & actual_positive).sum()

            wrong_mask = (predicted_positive != actual_positive).squeeze(1).cpu()
            if wrong_mask.any():
                wrong_imgs = X.cpu()[wrong_mask]
                wrong_true = actual_positive.cpu()[wrong_mask]
                wrong_pred = predicted_positive.cpu()[wrong_mask]
                wrong_prob = torch.sigmoid(logits.cpu())[wrong_mask]

                for img, t, p, pr in zip(wrong_imgs, wrong_true, wrong_pred, wrong_prob):
                    misclassified.append((
                        img.permute(1, 2, 0).numpy(),
                        int(t.item()),
                        int(p.item()),
                        pr.item(),
                    ))

    TP, FP, TN, FN = TP.item(), FP.item(), TN.item(), FN.item()

    precision = 100 * TP / (TP + FP)
    recall = 100 * TP / (TP + FN)
    accuracy = 100 * (TP + TN) / (TP + FP + TN + FN)

    F1 = 2 * precision * recall / (precision + recall)

    with open("metrics.txt", "a") as f:
        f.write(f"\nPrecision: {precision}%, Recall: {recall}%, F1: {F1}%, Accuracy: {accuracy}%")
        f.write("\n")
        f.write("\n                 Predicted Face | Predicted Not Face")
        f.write(f"\nActual Face          {TP}               {FN}")
        f.write(f"\nActual Not Face      {FP}               {TN}")

    cols = 6
    rows = max(1, math.ceil(len(misclassified) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 2.8 * rows))
    axes = axes.flatten()
    label_names = {1: "Face", 0: "Not Face"}

    for idx, ax in enumerate(axes):
        if idx < len(misclassified):
            img, true_y, pred_y, prob = misclassified[idx]
            ax.imshow(img.clip(0, 1))
            ax.set_title(
                f"True: {label_names[true_y]}\nPred: {label_names[pred_y]} ({prob:.2f})",
                fontsize=9,
                color="red",
            )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("misclassified_test.png", dpi=150)
    plt.show()