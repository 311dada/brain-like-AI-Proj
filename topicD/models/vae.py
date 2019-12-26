import torch
import torch.nn as nn
import numpy as np


class AutoEncoder(nn.Module):

    def __init__(self, encoder, decoder, **kwargs):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = getattr(nn, kwargs.get("reg_criterion", "MSELoss"))()

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def loss(self, x, return_recon=True):
        x_hat = self(x)

        if x.shape != x_hat.shape:
            x = x[:, :x_hat.size(1), :x_hat.size(-1)]
        if return_recon:
            return self.criterion(x, x_hat), x_hat
        else:
            return self.criterion(x, x_hat)



