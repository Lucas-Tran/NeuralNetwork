import json
from torch import nn

class ModelNN(nn.Module):
    def __init__(self, architecture_or_path):
        super(ModelNN, self).__init__()

        if isinstance(architecture_or_path, str):
            with open(architecture_or_path) as f:
                architecture = json.load(f)
        else:
            architecture = architecture_or_path

        n = architecture["n"]
        activations = architecture["activations"]
        l = len(activations)

        self.activations = nn.ModuleList()
        for i in range(l):
            match activations[i]:
                case "sigmoid":
                    self.activations.append(nn.Sigmoid())
                case "relu":
                    self.activations.append(nn.ReLU())
                case "tanh":
                    self.activations.append(nn.Tanh())
                case "softmax":
                    self.activations.append(nn.Softmax(dim=1))
                case "linear":
                    self.activations.append(nn.Identity())

        self.layers = nn.ModuleList()
        for i in range(l):
            self.layers.append(nn.Linear(n[i], n[i + 1]))

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x