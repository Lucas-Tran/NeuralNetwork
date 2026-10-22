import numpy as np
import matplotlib.pyplot as plt

import json
import json_parser as jp

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def forward(X, l, activations, params):
    w = params["w"]
    b = params["b"]
    cache = {"z": [], "a": []}
    a = X
    cache["a"].append(a)
    for i in range(l):
        z = np.dot(w[i], a) + b[i]
        match activations[i]:
            case "sigmoid":
                a = sigmoid(z)
            case "relu":
                a = relu(z)
            case "tanh":
                a = np.tanh(z)
        cache["z"].append(z)
        cache["a"].append(a)
    return a, cache


def compute_cost(M, X, Y_hat, l, activations, params):
    y, _ = forward(X, l, activations, params)
    losses = np.sum(np.square(Y_hat - y), axis=0)
    cost = np.sum(losses) / M
    return cost

def backward(M, X, Y_hat, l, activations, params):
    y, cache = forward(X, l, activations, params)

    grads = {"dw": [None] * l, "db": [None] * l}

    da = 2 * (y - Y_hat)
    for i in range(l - 1, -1, -1):
        a = cache["a"][i + 1]
        a_prev = cache["a"][i]
        z = cache["z"][i]

        match activations[i]:
            case "sigmoid":
                dz = da * a * (1 - a)
            case "relu":
                dz = da * (z >= 0).astype(float)
            case "tanh":
                dz = da * (1 - np.square(a))
            case _:
                raise ValueError(f"No activation named {activations[i]}")

        grads["dw"][i] = np.dot(dz, a_prev.T) / M
        grads["db"][i] = np.sum(dz, axis=1, keepdims=True) / M

        da = np.dot(params["w"][i].T, dz)

    return grads

def update_params(alpha, l, params, grads):
    w = params["w"]
    b = params["b"]
    dw = grads["dw"]
    db = grads["db"]
    for i in range(l):
        w[i] -= alpha * dw[i]
        b[i] -= alpha * db[i]
    return {"w": w, "b": b}

def gradient_decent(M, X, Y_hat, l, activations, alpha, iterations, params):
    training_iterations = []
    costs = []
    for i in range(iterations):
        current_cost = compute_cost(M, X, Y_hat, l, activations, params)
        assert (len(costs) == 0 or current_cost <= costs[-1]), f"Cost is increasing on iteration {i}"
        costs.append(current_cost)
        training_iterations.append(i)
        grads = backward(M, X, Y_hat, l, activations, params)
        params = update_params(alpha, l, params, grads)
    plt.plot(training_iterations, costs, marker='o', color='blue')
    plt.xlabel("Iteration Index (i)")
    plt.ylabel("Loss")
    plt.title("Loss Over Iterations")
    plt.show()
    return params

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
        params["w"].append(np.random.randn(n[i + 1], n[i]) * 0.1)
        params["b"].append(np.zeros((n[i + 1], 1)))


    params = gradient_decent(M, X, Y_hat, l, activations, alpha, num_iterations, params)
    with open("output.txt", "a") as file:
        print(f"Training set error: {compute_cost(M, X, Y_hat, l, activations, params)}", file=file)
        M_dev, X_dev, Y_dev = jp.parse_data("data/dev.json")

        print(f"Dev set error: {compute_cost(M_dev, X_dev, Y_dev, l, activations, params)}", file=file)