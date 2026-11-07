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
        self.activations = nn.ModuleList()
        self.poolings = nn.ModuleList()
        prev_channels = convolutional_layers["input_size"]["channels"]
        for out_channels, stride, kernel_size, padding, activation, pooling in zip(
                convolutional_layers["channels"],
                convolutional_layers["strides"],
                convolutional_layers["kernel_sizes"],
                convolutional_layers["paddings"],
                convolutional_layers["activations"],
                convolutional_layers["pooling"]):
            self.convolutions.append(nn.Conv2d(in_channels=prev_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size, padding=padding))
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
        for i in range(len(self.convolutions)):
            x = self.convolutions[i](x)
            x = self.activations[i](x)
            x = self.poolings[i](x)
        return x

    def forward(self, x):
        x = self.forward_convs(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x