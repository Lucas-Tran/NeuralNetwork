import matplotlib.pyplot as plt
import torch

def gradient_decent(model, train_loader, dev_loader, cost_func, hyperparameters):
    epoch_indices = []
    training_costs = []
    dev_costs = []

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using:", device)

    print("Training in progress...")

    model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["lr"], weight_decay=hyperparameters["weight_decay"])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hyperparameters["step_size"], gamma=hyperparameters["gamma"])
    for epoch in range(hyperparameters["epochs"]):
        model.train()

        train_cost = 0
        for X, Y in train_loader:
            X = X.to(device)
            Y = Y.to(device)
            # Clear gradients
            optimizer.zero_grad()

            # Get Cost (forward prop)
            prediction = model(X)
            current_cost = cost_func(prediction, Y)
            train_cost += current_cost.item() / len(train_loader)

            # Backward prop and gradient decent
            current_cost.backward()
            optimizer.step()
        scheduler.step()

        model.eval()

        dev_cost = 0
        with torch.no_grad():
            for (X, Y) in dev_loader:
                X = X.to(device)
                Y = Y.to(device)
                prediction = model(X)
                dev_cost += cost_func(prediction, Y).item() / len(dev_loader)

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