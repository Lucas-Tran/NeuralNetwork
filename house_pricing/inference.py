import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ModelNN import ModelNN
from select_device import select_device

if __name__ == "__main__":
    device = select_device()

    model = ModelNN("architecture.json")
    model.to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
    model.eval()

    print("Enter House data")
    print("----------------")
    while True:
        house_price = input("Enter House price in thousands of square feet (q to quit): ")
        if house_price == "q":
            break
        house_price = float(house_price)
        num_bedrooms = float(input("Enter number of bedrooms: "))
        num_bathrooms = float(input("Enter number of bathrooms: "))
        house_age = float(input("Enter house age in decades: "))

        house_data = torch.tensor([[house_price, num_bedrooms, num_bathrooms, house_age]], device=device)
        with torch.no_grad():
            house_price = model(house_data)

        print(f"House price: ${house_price.item() * 1000000}")