import matplotlib.pyplot as plt
import torch.nn as nn
import torch

def gradient_decent(model, X, Y_hat, alpha, iterations):
    training_iterations = []
    costs = []

    model.train()

    print("Training in progress...")

    cost_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
    for i in range(iterations):
        # Clear gradients
        optimizer.zero_grad()

        # Get Cost (forward prop)
        prediction = model(X)
        current_cost = cost_func(prediction, Y_hat)

        # Record cost
        costs.append(current_cost.item())
        training_iterations.append(i)

        # Backward prop and gradient decent
        current_cost.backward()
        optimizer.step()
    plt.plot(training_iterations, costs, marker='o', color='blue')
    plt.xlabel("Iteration Index (i)")
    plt.ylabel("Loss")
    plt.title("Loss Over Iterations")
    plt.show()