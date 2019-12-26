import torch
import torch.nn as nn
import numpy as np


class ConvDecoder(nn.Module):

    def __init__(self, num_freq, latent_dim, conv_input_shape, **kwargs):
        super(ConvDecoder, self).__init__()
        fc_nodes = kwargs.get("fc_nodes", [512]) + [np.prod(conv_input_shape[0])]  
        channels = kwargs.get("channels", [256, 128, 64]) + [1]
        kernel_sizes = kwargs.get("kernels", [(3, 1), (3, 1), (1, num_freq)])
        strides = kwargs.get("strides", [(2, 1), (2, 1), (1, 1)])
        self.conv_input_shape = conv_input_shape

        fc_layers = nn.ModuleList()
        for nl, (h_in, h_out) in enumerate(
                zip([latent_dim] + fc_nodes, fc_nodes)):
            fc_layers.append(nn.Linear(h_in, h_out))
            fc_layers.append(nn.BatchNorm1d(h_out))
            fc_layers.append(nn.ReLU())
        self.fc_layers = nn.Sequential(*fc_layers)

        conv_layers = nn.ModuleList()
        for nl, (c_in, c_out, k, s, t_in, t_out) in enumerate(
                zip(channels, channels[1:], kernel_sizes, strides,
                    conv_input_shape[2][1:][::-1], 
                    conv_input_shape[2][:-1][::-1])):
            if (t_in - 1) * s[0]  + k[0] != t_out:
                o_p = t_out - (t_in - 1) * s[0] - k[0]
            else:
                o_p = 0
            conv_layers.append(
                nn.ConvTranspose2d(c_in, c_out, k, s, output_padding=(o_p, 0)))
            if c_out != 1:
                conv_layers.append(nn.BatchNorm2d(c_out))
                conv_layers.append(nn.ReLU())
        self.conv_layers = nn.Sequential(*conv_layers)
        
        

    def forward(self, z):
        out = self.fc_layers(z)
        out = out.reshape((-1,) + self.conv_input_shape[0])
        out = self.conv_layers(out)
        # for layer in self.conv_layers:
            # out = layer(out)
            # print(out.shape)
        return out.squeeze(1)



class FcDecoder(nn.Module):

    def __init__(self, latent_dim, hid_dims, out_dim):
        super(FcDecoder, self).__init__()
        dims = [latent_dim] + hid_dims + [out_dim]
        layers = nn.ModuleList()
        for n_layer, (in_dim, o_dim) in enumerate(
            zip(dims, dims[1:])):
            layers.append(nn.Linear(in_dim, o_dim))
            if o_dim != out_dim:
                layers.append(nn.BatchNorm1d(o_dim))
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)


    def forward(self, x):
        return self.model(x)
