from torch.nn import CrossEntropyLoss

from torchvision import transforms
import ModelCNN
import gradient_decent as gd
import data_loader as dl

import gradio as gr

import json
import torch

transform_pipeline = transforms.Compose([
    transforms.RandomInvert(p=1.0),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])


chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZcdefghnqrt"

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
        predicted_class = int(torch.argmax(logits, dim=1).item())
        print(f"Predicted class: {predicted_class}")

    return f"Predicted Digit: {chars[predicted_class]}"


if __name__ == "__main__":
    model = ModelCNN.ModelCNN("EMNIST/architecture_CNN.json")

    with open("hyperparameters.json") as f:
        hyperparameters = json.load(f)

    if hyperparameters["retrain"]:

        train_loader, dev_loader = dl.load_data(hyperparameters["training"]["batch_size"])

        cost_func = CrossEntropyLoss()
        gd.gradient_decent(model, train_loader, dev_loader, cost_func, hyperparameters["training"])

        torch.save(model.state_dict(), "EMNIST/params.pth")

        model.eval()

        error = 0
        total = 0
        with torch.no_grad():
            for (images, labels) in train_loader:
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                error += (predictions != labels).sum().item()
                total += images.size(0)
        error /= total
        print(f"Training set error: {error * 100}%")

        error = 0
        total = 0
        with torch.no_grad():
            for (images, labels) in dev_loader:
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                error += (predictions != labels).sum().item()
                total += images.size(0)
        error /= total
        print(f"Dev set error: {error * 100}%")
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