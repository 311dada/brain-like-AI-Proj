import torch
import torch.nn as nn

class dnn_classifier(nn.Module):

    def __init__(self, input_dim, num_cls, hid_dims):
        super(dnn_classifier, self).__init__()
        dims = [input_dim] + hid_dims + [num_cls]
        layers = nn.ModuleList()
        for n_layer, (in_dim, out_dim) in enumerate(
            zip(dims, dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != num_cls:
                layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)


    def forward(self, x):
        return self.model(x)

