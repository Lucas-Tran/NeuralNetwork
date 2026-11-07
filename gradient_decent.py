import matplotlib.pyplot as plt
import torch.nn as nn
import torch

def gradient_decent(model, train_loader, cost_func, alpha, epochs):
    training_iterations = []
    costs = []

    model.train()

    print("Training in progress...")

    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
    for epoch in range(epochs):
        for X, Y in train_loader:
            # Clear gradients
            optimizer.zero_grad()

            # Get Cost (forward prop)
            prediction = model(X)
            current_cost = cost_func(prediction, Y)

            # Record cost
            # costs.append(current_cost.item())
            # training_iterations.append(i)

            # Backward prop and gradient decent
            current_cost.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} completed!")
    # plt.plot(training_iterations, costs, marker='o', color='blue')
    # plt.xlabel("Iteration Index (i)")
    # plt.ylabel("Loss")
    # plt.title("Loss Over Iterations")
    # plt.show()