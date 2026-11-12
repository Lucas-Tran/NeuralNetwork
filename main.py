from torch.nn import CrossEntropyLoss

from torchvision import transforms
import ModelCNN
import gradient_decent as gd
import data_loader as dl

import gradio as gr

import json
import time
import torch

transform_pipeline = transforms.Compose([
    transforms.RandomInvert(p=1.0),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])


chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

def predict_drawing(sketch_dict):
    """Processes the sketch from the browser canvas and outputs predictions."""
    if sketch_dict is None:
        return "Please draw something on the canvas!"

    # Gradio sketchpad returns a dictionary containing a 'composite' image layer
    pil_image = sketch_dict["composite"]

    # Handle color channels based on your model input configuration
    pil_image = pil_image.convert("L")  # Convert to Grayscale

    # Apply sizing and normalization transforms, then add the Batch Dimension [1, C, H, W]
    model_device = next(model.parameters()).device
    tensor_image = transform_pipeline(pil_image).unsqueeze(0).to(model_device)
    # Run prediction safely without gradient computations
    with torch.no_grad():
        logits = model(tensor_image)
        print(logits)
        # Find the highest raw score index column
        predicted_class = int(torch.argmax(logits, dim=1).item())
        print(f"Predicted class: {predicted_class}")

    return f"Predicted Digit: {chars[predicted_class]}"

def calculate_error(dataset):
    errors = torch.zeros((), device=device, dtype=torch.long)
    total = 0
    with torch.no_grad():
        for (images, labels) in dataset:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            errors += (predictions != labels).sum()
            total += images.size(0)
    return 100 * errors.item() / total


if __name__ == "__main__":
    print("Initializing Model...")
    model = ModelCNN.ModelCNN("EMNIST/architecture_CNN.json")

    with open("hyperparameters.json") as f:
        hyperparameters = json.load(f)

    if hyperparameters["retrain"]:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using: {device} ({torch.cuda.get_device_name(device)})")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using:", device)
        else:
            device = torch.device("cpu")
            print("Using:", device)

        print("Loading Data...")
        training_config = hyperparameters["training"]
        train_loader, dev_loader = dl.load_data(
            training_config["batch_size"],
            num_workers=training_config.get("num_workers", 4),
            pin_memory=device.type == "cuda",
        )

        print("Training in progress...")

        model.to(device)

        cost_func = CrossEntropyLoss()
        gd.gradient_decent(device, model, train_loader, dev_loader, cost_func, hyperparameters["training"])

        torch.save(model.state_dict(), "EMNIST/params.pth")

        model.eval()

        print(f"Train set error: {calculate_error(train_loader)}%")
        print(f"Dev set error: {calculate_error(dev_loader)}%")

    else:
        model.load_state_dict(torch.load("EMNIST/params.pth", weights_only=True))
        model.eval()


    interface = gr.Interface(
        fn=predict_drawing,
        inputs=gr.Sketchpad(type="pil", label="Draw a Number Here"),
        outputs=gr.Textbox(label="Model Output Decision"),
        title="Handwritten Digit Draw & Predict",
        description="Sketch any single digit number inside the box and click submit to watch your dynamic CNN classify it instantly!"
    )

    interface.launch(server_name="0.0.0.0", server_port=7860)