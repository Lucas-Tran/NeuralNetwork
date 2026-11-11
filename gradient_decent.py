from contextlib import nullcontext
from time import perf_counter

import matplotlib.pyplot as plt
import torch


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def gradient_decent(device, model, train_loader, dev_loader, cost_func, hyperparameters):
    epoch_indices = []
    training_costs = []
    dev_costs = []


    use_cuda = device.type == "cuda"
    compute_capability = torch.cuda.get_device_capability(device) if use_cuda else None
    use_bf16 = use_cuda and compute_capability[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda and not use_bf16)

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
        fused=use_cuda,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hyperparameters["step_size"], gamma=hyperparameters["gamma"])
    for epoch in range(hyperparameters["epochs"]):
        _synchronize(device)
        epoch_started = perf_counter()
        model.train()

        train_loss_total = torch.zeros((), device=device)
        train_data_wait = 0.0
        train_started = perf_counter()
        train_iterator = iter(train_loader)
        while True:
            batch_started = perf_counter()
            try:
                X, Y = next(train_iterator)
            except StopIteration:
                break
            train_data_wait += perf_counter() - batch_started

            X = X.to(device, non_blocking=use_cuda)
            Y = Y.to(device, non_blocking=use_cuda)
            optimizer.zero_grad(set_to_none=True)

            autocast = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_cuda else nullcontext()
            with autocast:
                prediction = model(X)
                current_cost = cost_func(prediction, Y)
            train_loss_total += current_cost.detach()

            if scaler.is_enabled():
                scaler.scale(current_cost).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                current_cost.backward()
                optimizer.step()
        scheduler.step()
        _synchronize(device)
        train_elapsed = perf_counter() - train_started
        train_cost = (train_loss_total / len(train_loader)).item()

        model.eval()

        dev_loss_total = torch.zeros((), device=device)
        dev_data_wait = 0.0
        dev_model_elapsed = 0.0
        dev_model_events = []
        dev_started = perf_counter()
        with torch.no_grad():
            dev_iterator = iter(dev_loader)
            while True:
                batch_started = perf_counter()
                try:
                    X, Y = next(dev_iterator)
                except StopIteration:
                    break
                dev_data_wait += perf_counter() - batch_started

                X = X.to(device, non_blocking=use_cuda)
                Y = Y.to(device, non_blocking=use_cuda)
                autocast = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_cuda else nullcontext()
                with autocast:
                    if device.type == "cuda":
                        model_started = torch.cuda.Event(enable_timing=True)
                        model_finished = torch.cuda.Event(enable_timing=True)
                    elif device.type == "mps":
                        model_started = torch.mps.Event(enable_timing=True)
                        model_finished = torch.mps.Event(enable_timing=True)
                    else:
                        model_started_at = perf_counter()
                    if device.type in ("cuda", "mps"):
                        model_started.record()
                    prediction = model(X)
                    if device.type in ("cuda", "mps"):
                        model_finished.record()
                        dev_model_events.append((model_started, model_finished))
                    else:
                        dev_model_elapsed += perf_counter() - model_started_at
                    dev_loss_total += cost_func(prediction, Y)
        _synchronize(device)
        if dev_model_events:
            dev_model_elapsed = sum(
                started.elapsed_time(finished) for started, finished in dev_model_events
            ) / 1000
        dev_elapsed = perf_counter() - dev_started
        dev_cost = (dev_loss_total / len(dev_loader)).item()
        epoch_elapsed = perf_counter() - epoch_started

        # Record Epoch
        epoch_indices.append(epoch)
        training_costs.append(train_cost)
        dev_costs.append(dev_cost)
        print(
            f"Epoch {epoch + 1}/{hyperparameters['epochs']} | "
            f"train loop: {train_elapsed:.2f}s (batch wait: {train_data_wait:.2f}s) | "
            f"validation: {dev_elapsed:.2f}s (batch wait: {dev_data_wait:.2f}s, "
            f"model(X): {dev_model_elapsed:.3f}s) | "
            f"total: {epoch_elapsed:.2f}s | "
            f"train loss: {train_cost:.4f} | dev loss: {dev_cost:.4f}"
        )

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