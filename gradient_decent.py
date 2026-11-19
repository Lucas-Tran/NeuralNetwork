from contextlib import nullcontext
from time import perf_counter

try:
    # PyTorch >= 2.3
    from torch.amp import GradScaler
except ImportError:
    # PyTorch < 2.3
    from torch.cuda.amp import GradScaler

import matplotlib.pyplot as plt
import torch
import os

def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def gradient_decent(device, model, train_loader, dev_loader, cost_func, hyperparameters, error_func=None, augmentation=None):
    epoch_indices = []
    training_costs = []
    dev_costs = []

    training_started = perf_counter()

    use_cuda = device.type == "cuda"
    compute_capability = torch.cuda.get_device_capability(device) if use_cuda else None
    use_bf16 = use_cuda and compute_capability[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = GradScaler(enabled=use_cuda and not use_bf16)

    if use_cuda:
        precision = "BF16" if use_bf16 else "FP16"
        print(f"Using {precision} autocast on CUDA compute capability {compute_capability[0]}.{compute_capability[1]}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparameters["lr"],
        weight_decay=hyperparameters["weight_decay"],
        fused=use_cuda
    )
    best_cost = float("inf")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hyperparameters["step_size"], gamma=hyperparameters["gamma"])
    for epoch in range(hyperparameters["epochs"]):
        _synchronize(device)
        model.train()

        train_loss_total = torch.zeros((), device=device)
        for X, Y in train_loader:

            X = X.to(device, non_blocking=use_cuda)
            Y = Y.to(device, non_blocking=use_cuda)
            if augmentation is not None:
                X = augmentation(X)
            optimizer.zero_grad(set_to_none=True)

            autocast = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_cuda else nullcontext()
            with autocast:
                prediction = model(X)
                current_cost = cost_func(prediction, Y)
            train_loss_total += current_cost.detach() * X.size(0)

            if scaler.is_enabled():
                scaler.scale(current_cost).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                current_cost.backward()
                optimizer.step()
        scheduler.step()
        _synchronize(device)
        train_cost = (train_loss_total / len(train_loader.dataset)).item()

        model.eval()

        dev_loss_total = torch.zeros((), device=device)
        with torch.no_grad():
            for X, Y in dev_loader:
                X = X.to(device, non_blocking=use_cuda)
                Y = Y.to(device, non_blocking=use_cuda)
                autocast = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_cuda else nullcontext()
                with autocast:
                    prediction = model(X)
                    dev_loss_total += cost_func(prediction, Y) * X.size(0)
        if error_func is not None:
            error_rate = error_func(dev_loader, model, device)

        dev_cost = (dev_loss_total / len(dev_loader.dataset)).item()
        if dev_cost < best_cost:
            best_cost = dev_cost
            torch.save(model.state_dict(), "tmp.pth")
            print(f"Model saved!")
        _synchronize(device)
        
        # Record Epoch
        epoch_indices.append(epoch)
        training_costs.append(train_cost)
        dev_costs.append(dev_cost)
        print(f"Epoch {epoch + 1}/{hyperparameters['epochs']}")
        if error_func is not None:
            print(f"train loss: {train_cost:.4f} | dev loss: {dev_cost:.4f} | error rate: {error_rate:.2f}%")
        else:
            print(f"train loss: {train_cost:.4f} | dev loss: {dev_cost:.4f}")

    
    model.load_state_dict(torch.load("tmp.pth", weights_only=True))
    os.remove(os.path.join("tmp.pth"))

    training_elapsed = perf_counter() - training_started
    print(
        f"Training and validation time: {training_elapsed / 60:.2f} minutes ({training_elapsed:.1f} seconds)")

    plt.figure(figsize=(10, 6))

    plt.plot(epoch_indices, training_costs, label="Training Cost", color="blue", marker="o", linestyle="-")
    plt.plot(epoch_indices, dev_costs, label="Dev Cost", color="red", marker="o", linestyle="-")

    plt.xlabel("Epoch")
    plt.ylabel("Average Loss (Cross Entropy)")
    plt.title("Training/Dev Set Loss")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    plt.show()