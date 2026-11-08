from torch.nn import CrossEntropyLoss

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import ModelCNN
import gradient_decent as gd

import gradio as gr
from PIL import Image

import json
import torch

def load_data():
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=15),  # Rotates up to 15°
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Shifts up to 10% x/y
        transforms.ToTensor()
    ])

    test_transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparameters["training"]["batch_size"],
        shuffle=True
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    return train_dataset, train_loader, test_dataset


def predict_drawing(sketch_dict):
    """Processes the sketch from the browser canvas and outputs predictions."""
    if sketch_dict is None:
        return "Please draw something on the canvas!"

    # Gradio sketchpad returns a dictionary containing a 'composite' image layer
    pil_image = sketch_dict["composite"]

    # Handle color channels based on your model input configuration
    pil_image = pil_image.convert("L")  # Convert to Grayscale

    # Apply sizing and normalization transforms, then add the Batch Dimension [1, C, H, W]
    tensor_image = transform_pipeline(pil_image).unsqueeze(0)

    # Run prediction safely without gradient computations
    with torch.no_grad():
        logits = model(tensor_image)
        print(logits)
        # Find the highest raw score index column
        predicted_class = torch.argmax(logits, dim=1).item()

    return f"Predicted Digit: {predicted_class}"

if __name__ == "__main__":

    with open("hyperparameters.json") as f:
        hyperparameters = json.load(f)

    model = ModelCNN.ModelCNN("NNArchitecture/CNN.json")

    if hyperparameters["retrain"]:

        train_dataset, train_loader, test_dataset = load_data()

        alpha = hyperparameters["training"]["alpha"]
        num_iterations = hyperparameters["training"]["epochs"]
        step_size = hyperparameters["training"]["step_size"]
        gamma = hyperparameters["training"]["gamma"]



        cost_func = CrossEntropyLoss()
        gd.gradient_decent(model, train_loader, cost_func, alpha, num_iterations, step_size, gamma)

        torch.save(model.state_dict(), "params_numbers.pth")

        model.eval()
        with open("output.txt", "a") as file:
            images = train_dataset.data.float().unsqueeze(1) / 255
            labels = train_dataset.targets

            with torch.no_grad():
                predictions = model(images)
            print(f"Training set error: {cost_func(predictions, labels).item()}", file=file)


            images = test_dataset.data.float().unsqueeze(1) / 255
            labels = test_dataset.targets

            with torch.no_grad():
                predictions = model(images)

            print(f"Dev set error: {cost_func(predictions, labels)}", file=file)
    else:
        model.load_state_dict(torch.load("params_numbers.pth", weights_only=True))
        model.eval()

    transform_pipeline = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.RandomInvert(p=1.0)
    ])

    interface = gr.Interface(
        fn=predict_drawing,
        inputs=gr.Sketchpad(type="pil", label="Draw a Number Here"),
        outputs=gr.Textbox(label="Model Output Decision"),
        title="Handwritten Digit Draw & Predict",
        description="Sketch any single digit number inside the box and click submit to watch your dynamic CNN classify it instantly!"
    )

    interface.launch()