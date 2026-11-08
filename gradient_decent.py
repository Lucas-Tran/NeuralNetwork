import matplotlib.pyplot as plt
import torch.nn as nn
import torch

def gradient_decent(model, train_loader, cost_func, alpha, epochs, step_size, gamma):
    training_iterations = []
    costs = []

    model.train()

    print("Training in progress...")

    optimizer = torch.optim.Adam(model.parameters(), lr=alpha, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    for epoch in range(epochs):
        cost = 0
        for X, Y in train_loader:
            # Clear gradients
            optimizer.zero_grad()

            # Get Cost (forward prop)
            prediction = model(X)
            current_cost = cost_func(prediction, Y)
            cost += current_cost.item()

            # Backward prop and gradient decent
            current_cost.backward()
            optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch + 1} completed!")
        print(f"Cost: {cost / len(train_loader)}")