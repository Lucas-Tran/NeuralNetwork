import torch
from torch import nn

class ModelNN(nn.Module):
    def __init__(self, n, activations):
        super(ModelNN, self).__init__()
        self.activations = nn.ModuleList()
        for i in range(len(n) - 1):
            match activations[i]:
                case "sigmoid":
                    self.activations.append(nn.Sigmoid())
                case "relu":
                    self.activations.append(nn.ReLU())
                case "tanh":
                    self.activations.append(nn.Tanh())
                case "linear":
                    self.activations.append(nn.Identity())

        self.layers = nn.ModuleList()
        for i in range(len(n) - 1):
            self.layers.append(nn.Linear(n[i], n[i + 1]))

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x