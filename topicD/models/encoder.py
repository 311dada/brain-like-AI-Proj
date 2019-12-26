import torch
import torch.nn as nn
import numpy as np


class ConvEncoder(nn.Module):
    
    def __init__(self, num_freq, latent_dim, **kwargs):
        super(ConvEncoder, self).__init__()
        self.num_freq = num_freq
        self.num_frame = kwargs.get("num_frame", 101)
        channels = kwargs.get("channels", [64, 128, 256])
        kernel_sizes = kwargs.get("kernels", [(1, num_freq), (3, 1), (3, 1)])
        strides = kwargs.get("strides", [(1, 1), (2, 1), (2, 1)])
        fc_nodes = kwargs.get("fc_nodes", [512])

        conv_layers = nn.ModuleList()
        for nl, (c_in, c_out, k, s) in enumerate(
                zip([1] + channels, channels, kernel_sizes, strides)):
            conv_layers.append(nn.Conv2d(c_in, c_out, k, s))
            conv_layers.append(nn.BatchNorm2d(c_out))
            conv_layers.append(nn.ReLU())
        self.conv_layers = nn.Sequential(*conv_layers)

        _, conv_size, _ = self.calculate_conv_output_shape()

        fc_layers = nn.ModuleList()
        for nl, (h_in, h_out) in enumerate(
                zip([conv_size] + fc_nodes, fc_nodes)):
            fc_layers.append(nn.Linear(h_in, h_out))
            if h_out != latent_dim:
                fc_layers.append(nn.BatchNorm1d(h_out))
                fc_layers.append(nn.ReLU())
        self.fc_layers = nn.Sequential(*fc_layers)

        self.outputlayer = nn.Linear(fc_nodes[-1], latent_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        hidden = self.fc_layers(x)
        return self.outputlayer(hidden)


    def calculate_conv_output_shape(self):
        frame_seq = [self.num_frame]
        with torch.no_grad():
            x = torch.randn(1, 1, self.num_frame, self.num_freq)
            for layer in self.conv_layers:
                x = layer(x)
                if isinstance(layer, nn.Conv2d):
                    frame_seq.append(x.shape[2])
                    # print(frame_seq[-1])
        return tuple(x.shape[1:]), x.reshape(1, -1).size(1), frame_seq


class FcEncoder(nn.Module):

    def __init__(self, input_dim, latent_dim, hid_dims):
        super(FcEncoder, self).__init__()
        dims = [input_dim] + hid_dims + [latent_dim]
        layers = nn.ModuleList()
        for n_layer, (in_dim, out_dim) in enumerate(
            zip(dims, dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != latent_dim:
                layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


