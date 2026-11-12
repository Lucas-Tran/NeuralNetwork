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
        dropout = architecture.get("dropout", [0.0] * l)


        self.layers = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(l):
            is_last_layer = (i == l - 1)

            self.layers.append(nn.Linear(n[i], n[i + 1], bias=is_last_layer))

            if is_last_layer:
                self.batch_norm.append(nn.Identity())
            else:
                self.batch_norm.append(nn.BatchNorm1d(n[i + 1]))

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

            self.dropouts.append(nn.Dropout(p=dropout[i]))

    def forward(self, x):
        for layer, batch_norm, activation, dropout in zip(self.layers, self.batch_norm, self.activations, self.dropouts):
            x = layer(x)
            x = batch_norm(x)
            x = activation(x)
            x = dropout(x)
        return x