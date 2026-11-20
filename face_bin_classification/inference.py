import torch
import cv2
from torchvision import transforms


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ModelCNN import ModelCNN
from select_device import select_device

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def predict(frame):
    # Convert BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to tensor and add batch dimension
    image = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(image)
        probability = torch.sigmoid(logits).item()

    prediction = "Face" if probability >= 0.5 else "Not Face"

    return prediction, probability

if __name__ == '__main__':
    device = select_device()

    model = ModelCNN("architecture.json")
    model.to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))

    model.eval()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    try:
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            if not ret:
                break

            prediction, probability = predict(frame)

            text = f"{prediction}: {probability:.2%}"

            cv2.putText(
                frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Face Classifier", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()