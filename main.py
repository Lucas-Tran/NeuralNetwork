import numpy as np
import matplotlib.pyplot as plt

import json
import json_parser as jp
import torch

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def forward(X, l, activations, params):
    w = params["w"]
    b = params["b"]
    a = X
    for i in range(l):
        z = torch.matmul(w[i], a) + b[i]
        match activations[i]:
            case "sigmoid":
                a = torch.sigmoid(z)
            case "relu":
                a = torch.relu(z)
            case "tanh":
                a = torch.tanh(z)
            case "linear":
                a = z
    return a


def compute_cost(M, X, Y_hat, l, activations, params):
    y = forward(X, l, activations, params)
    return torch.mean((Y_hat - y) ** 2)

def gradient_decent(M, X, Y_hat, l, activations, alpha, iterations, params):
    training_iterations = []
    costs = []
    optimizer = torch.optim.Adam(params["w"] + params["b"],lr=alpha)
    for i in range(iterations):
        optimizer.zero_grad()
        current_cost = compute_cost(M, X, Y_hat, l, activations, params)

        costs.append(current_cost.item())
        training_iterations.append(i)

        current_cost.backward()
        optimizer.step()
    plt.plot(training_iterations, costs, marker='o', color='blue')
    plt.xlabel("Iteration Index (i)")
    plt.ylabel("Loss")
    plt.title("Loss Over Iterations")
    plt.show()
    return params

def predict(house_size, bedrooms, bathrooms, house_age,
            params, activations, mean, std):

    X = torch.tensor([
        [house_size],
        [bedrooms],
        [bathrooms],
        [house_age]
    ], dtype=torch.float32)

    X = (X - mean) / std

    with torch.no_grad():
        prediction = forward(
            X,
            len(params["w"]),
            activations,
            params
        )

    return prediction.item() * 1_000_000

if __name__ == "__main__":
    M, X, Y_hat = jp.parse_data("data/train.json")

    with open("hyperparameters.json") as f:
        data = json.load(f)
        alpha = data["training"]["alpha"]
        num_iterations = data["training"]["num_iterations"]
        n = data["network-architecture"]["n"]
        l = len(n) - 1
        activations = data["network-architecture"]["activations"]
    params = {"w": [], "b": []}

    for i in range(l):
        params["w"].append(torch.nn.init.xavier_uniform_(
        torch.empty(n[i + 1], n[i])
    ).requires_grad_())
        params["b"].append(torch.zeros(n[i + 1], 1, requires_grad=True))

    mean = X.mean(dim=1, keepdim=True)
    std = X.std(dim=1, keepdim=True, correction=0)

    X_normalized = (X - mean) / std


    params = gradient_decent(M, X_normalized, Y_hat, l, activations, alpha, num_iterations, params)

    with open("output.txt", "a") as file:
        print(f"Training set error: {compute_cost(M, X_normalized, Y_hat, l, activations, params)}", file=file)
        M_dev, X_dev, Y_dev = jp.parse_data("data/dev.json")

        X_dev = (X_dev - mean) / std

        with torch.no_grad():
            dev_predictions = forward(
                X_dev, l, activations, params
            )

        print(f"Dev set error: {compute_cost(M_dev, X_dev, Y_dev, l, activations, params)}", file=file)

    while True:
        print("\nEnter house information (or type 'q' to quit)")

        house_size = input("House size (thousands of sq ft): ")

        if house_size.lower() == "q":
            break

        bedrooms = float(input("Number of bedrooms: "))
        bathrooms = float(input("Number of bathrooms: "))
        house_age = float(input("House age: "))

        price = predict(
            float(house_size),
            bedrooms,
            bathrooms,
            house_age,
            params,
            activations,
            mean,
            std
        )

        print(f"Predicted price: ${price:,.2f}")