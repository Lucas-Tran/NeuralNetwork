import torch.nn as nn
import torch
import json
import ModelNN

class ModelCNN(nn.Module):
    def __init__(self, architecture_json_path):
        super(ModelCNN, self).__init__()
        with open(architecture_json_path) as f:
            architecture = json.load(f)

        convolutional_layers = architecture["convolutional_layers"]

        self.convolutions = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.poolings = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        prev_channels = convolutional_layers["input_size"]["channels"]
        for out_channels, stride, kernel_size, padding, activation, pooling, dropout in zip(
                convolutional_layers["channels"],
                convolutional_layers["strides"],
                convolutional_layers["kernel_sizes"],
                convolutional_layers["paddings"],
                convolutional_layers["activations"],
                convolutional_layers["pooling"],
                convolutional_layers.get("dropout", [0.0] * len(convolutional_layers["channels"])),):
            self.convolutions.append(nn.Conv2d(in_channels=prev_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size, padding=padding, bias=False))
            self.batch_norm.append(nn.BatchNorm2d(num_features=out_channels))
            match activation:
                case "relu":
                    self.activations.append(nn.ReLU())
                case "tanh":
                    self.activations.append(nn.Tanh())
                case "sigmoid":
                    self.activations.append(nn.Sigmoid())
                case "linear":
                    self.activations.append(nn.Identity())
            if pooling == None:
                self.poolings.append(nn.Identity())
            else:
                match pooling["type"]:
                    case "max":
                        self.poolings.append(nn.MaxPool2d(kernel_size=pooling["kernel_size"], stride=pooling["stride"]))
                    case "average":
                        self.poolings.append(nn.AvgPool2d(kernel_size=pooling["kernel_size"], stride=pooling["stride"]))
            self.dropouts.append(nn.Dropout2d(p=dropout))
            prev_channels = out_channels

        FCLs = architecture["FCLs"]
        FCLs["n"].insert(0, self._get_conv_output_shape(convolutional_layers["input_size"]["channels"], convolutional_layers["input_size"]["height"], convolutional_layers["input_size"]["width"]))

        self.fully_connected = ModelNN.ModelNN(FCLs)

    def _get_conv_output_shape(self, channels, height, width):
        self.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, height, width)
            dummy_output = self.forward_convs(dummy_input)
            return dummy_output.numel()

    def forward_convs(self, x):
        for convolution, bn, activation, pooling, dropout in zip(
                self.convolutions,
                self.batch_norm,
                self.activations,
                self.poolings,
                self.dropouts
        ):
            x = convolution(x)
            x = bn(x)
            x = activation(x)
            x = pooling(x)
            x = dropout(x)
        return x

    def forward(self, x):
        x = self.forward_convs(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fully_connected(x)
        return x