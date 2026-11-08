import matplotlib.pyplot as plt
import torch

def gradient_decent(model, dev_dataset, train_loader, cost_func, hyperparameters):
    epoch_indices = []
    training_costs = []
    dev_costs = []

    print("Training in progress...")


    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["lr"], weight_decay=hyperparameters["weight_decay"])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hyperparameters["step_size"], gamma=hyperparameters["gamma"])
    for epoch in range(hyperparameters["epochs"]):
        train_cost = 0
        model.train()
        for X, Y in train_loader:
            # Clear gradients
            optimizer.zero_grad()

            # Get Cost (forward prop)
            prediction = model(X)
            current_cost = cost_func(prediction, Y)
            train_cost += current_cost.item()

            # Backward prop and gradient decent
            current_cost.backward()
            optimizer.step()
        scheduler.step()

        images = dev_dataset.data.float().unsqueeze(1) / 255
        labels = dev_dataset.targets

        with torch.no_grad():
            dev_cost = cost_func(model(images), labels).item()
        train_cost /= len(train_loader)
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