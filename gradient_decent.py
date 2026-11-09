from contextlib import nullcontext

import matplotlib.pyplot as plt
import torch

def gradient_decent(device, model, train_loader, dev_loader, cost_func, hyperparameters):
    epoch_indices = []
    training_costs = []
    dev_costs = []


    use_cuda = device.type == "cuda"
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and not use_bf16)

    if use_cuda:
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
        model.train()

        train_loss_total = torch.zeros((), device=device)
        for X, Y in train_loader:
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
        train_cost = (train_loss_total / len(train_loader)).item()

        model.eval()

        dev_loss_total = torch.zeros((), device=device)
        with torch.no_grad():
            for (X, Y) in dev_loader:
                X = X.to(device, non_blocking=use_cuda)
                Y = Y.to(device, non_blocking=use_cuda)
                autocast = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_cuda else nullcontext()
                with autocast:
                    prediction = model(X)
                    dev_loss_total += cost_func(prediction, Y)
        dev_cost = (dev_loss_total / len(dev_loader)).item()

        # Record Epoch
        epoch_indices.append(epoch)
        training_costs.append(train_cost)
        dev_costs.append(dev_cost)
        print(f"Epoch {epoch + 1} completed!")
        print(f"Train Cost: {train_cost} | Dev Cost: {dev_cost}")

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