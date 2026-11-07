import ModelNN
import gradient_decent as gd

import json
import json_parser as jp
import torch
import torch.nn as nn

if __name__ == "__main__":
    M, X, Y_hat = jp.parse_data("data/train.json")

    with open("hyperparameters.json") as f:
        data = json.load(f)
        alpha = data["training"]["alpha"]
        num_iterations = data["training"]["num_iterations"]
        n = data["network-architecture"]["n"]
        activations = data["network-architecture"]["activations"]
        retrain = data["retrain"]

    model = ModelNN.ModelNN(n, activations)

    if retrain:
        gd.gradient_decent(model, X, Y_hat, alpha, num_iterations)

        torch.save(model.state_dict(), "params.pth")

        model.eval()
        with open("output.txt", "a") as file:
            cost_func = nn.MSELoss()
            print(f"Training set error: {cost_func(model(X), Y_hat).item()}", file=file)
            M_dev, X_dev, Y_dev = jp.parse_data("data/dev.json")

            print(f"Dev set error: {cost_func(model(X_dev), Y_dev).item()}", file=file)
    else:
        model.load_state_dict(torch.load("params.pth", weights_only=True))
        model.eval()

    while True:
        print("\nEnter house information (or type 'q' to quit)")

        house_size = input("House size (thousands of sq ft): ")

        if house_size.lower() == "q":
            break

        house_size = float(house_size)
        bedrooms = float(input("Number of bedrooms: "))
        bathrooms = float(input("Number of bathrooms: "))
        house_age = float(input("House age: "))

        price = model(torch.tensor([[house_size, bedrooms, bathrooms, house_age]])).item() * 1000000

        print(f"Predicted price: ${price:,.2f}")