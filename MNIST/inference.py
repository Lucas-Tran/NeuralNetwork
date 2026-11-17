from torchvision import transforms
import gradio as gr

import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ModelCNN import ModelCNN
from select_device import select_device

transform_pipeline = transforms.Compose([
    transforms.RandomInvert(p=1.0),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

def predict_drawing(sketch_dict):
    """Processes the sketch from the browser canvas and outputs predictions."""
    if sketch_dict is None:
        return "Please draw something on the canvas!"

    pil_image = sketch_dict["composite"]

    pil_image = pil_image.convert("L")

    tensor_image = transform_pipeline(pil_image).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor_image)
        predicted_class = torch.argmax(logits, dim=1).item()

    return f"Predicted Digit: {predicted_class}"


if __name__ == "__main__":
    device = select_device()
    model = ModelCNN("architecture.json")
    model.to(device)
    model.load_state_dict(torch.load("model.pth", weights_only=True, map_location=device))
    model.eval()


    interface = gr.Interface(
        fn=predict_drawing,
        inputs=gr.Sketchpad(type="pil", label="Draw a Number Here"),
        outputs=gr.Textbox(label="Model Output Decision"),
        title="Handwritten Digit Draw & Predict",
        description="Write a digit and click as the ModelCNN classifies it!"
    )

    interface.launch(server_name="0.0.0.0", server_port=7860)