from torch.nn import CrossEntropyLoss

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import ModelCNN
import gradient_decent as gd

import json
import torch

if __name__ == "__main__":

    with open("hyperparameters.json") as f:
        hyperparameters = json.load(f)

    model = ModelCNN.ModelCNN("NNArchitecture/CNN.json")

    if hyperparameters["retrain"]:

        transform = transforms.ToTensor()

        train_dataset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )

        alpha = hyperparameters["training"]["alpha"]
        num_iterations = hyperparameters["training"]["epochs"]

        train_loader = DataLoader(
            train_dataset,
            batch_size=hyperparameters["training"]["batch_size"],
            shuffle=True
        )

        cost_func = CrossEntropyLoss()
        gd.gradient_decent(model, train_loader, cost_func, alpha, num_iterations)

        torch.save(model.state_dict(), "params_numbers.pth")

        model.eval()
        with open("output.txt", "a") as file:
            images = train_dataset.data.float().unsqueeze(1) / 255
            labels = train_dataset.targets

            with torch.no_grad():
                predictions = model(images)
            print(f"Training set error: {cost_func(predictions, labels).item()}", file=file)



            test_dataset = datasets.MNIST(
                root="./data",
                train=False,
                download=True,
                transform=transform
            )

            images = test_dataset.data.float().unsqueeze(1) / 255
            labels = test_dataset.targets

            with torch.no_grad():
                predictions = model(images)

            print(f"Dev set error: {cost_func(predictions, labels)}", file=file)
    else:
        model.load_state_dict(torch.load("params_numbers.pth", weights_only=True))
        model.eval()

    # while True:
    #     print("\nEnter house information (or type 'q' to quit)")
    #
    #     house_size = input("House size (thousands of sq ft): ")
    #
    #     if house_size.lower() == "q":
    #         break
    #
    #     house_size = float(house_size)
    #     bedrooms = float(input("Number of bedrooms: "))
    #     bathrooms = float(input("Number of bathrooms: "))
    #     house_age = float(input("House age: "))
    #
    #     price = model(torch.tensor([[house_size, bedrooms, bathrooms, house_age]])).item() * 1000000
    #
    #     print(f"Predicted price: ${price:,.2f}")