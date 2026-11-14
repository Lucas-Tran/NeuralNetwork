import torch


def calculate_error_multiple_classes(dataset, model, device):
    model.eval()
    errors = torch.zeros((), device=device, dtype=torch.long)
    total = 0
    with torch.no_grad():
        for (images, labels) in dataset:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            errors += (predictions != labels).sum()
            total += images.size(0)
    return 100 * errors.item() / total


def calculate_error_binary(dataset, model, device):
    model.eval()
    errors = torch.zeros((), device=device, dtype=torch.long)
    total = 0
    with torch.no_grad():
        for (images, labels) in dataset:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            outputs = model(images)
            predictions = outputs > 0
            errors += (predictions != labels).sum()
            total += images.size(0)
    return 100 * errors.item() / total