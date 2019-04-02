#!/usr/bin/env python

import torch

from torch import nn
from functools import reduce
from operator import mul


class Flatten(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        
    def forward(self, x):
        return x.view(-1, self.n)


class SlimAlexNet(nn.Module):
    """
    AlexNet with the classifier layers (fully-connected Linear layers) removed.
    """

    MAX_POOL_LAYERS = [3, 6, 13]
    # the shape of the MaxPool2D layer can be inspected using the following code
    # snipppet:

    # from torchsummary import summary
    # summary(torchvision.models.alexnet(pretrained=True), (3, 224, 224))
    MAX_POOL_OUT_SHAPE = [
        (64, 27, 27),
        (192, 13, 13),
        (256, 6, 6)
    ]

    def __init__(self, alexnet, max_pool_layer_index=1, dropout_ratio=0.5, last_layer_num_params=180):
        """
        Construct a slim versino of AlexNet by only reusing the first a few layers.
        The last layer maps the output of the MaxPool2D layer to 180 classes corresponding
        to 180 degrees.

        alexnet: pretrained AlexNet model.
        max_pool_layer_index: Select which maximum pooling layer to use (1, 2 or 3).
        dropout_ratio: The dropout ratio/proportion for the Dropout layer.
        """
        assert 1 <= max_pool_layer_index <= 3, \
            f"Invalid max_pool_layer_index {max_pool_layer_index}"
        
        super().__init__()

        # extract the first a few layers, then combine it with two extra layers
        # (1) a flattening layer that returns an 1d array
        # (2) a linear layer 
        layers_to_stack = self.MAX_POOL_LAYERS[max_pool_layer_index - 1]
        maxpool_out_shape = self.MAX_POOL_OUT_SHAPE[max_pool_layer_index - 1]

        num_params = reduce(mul, maxpool_out_shape)

        self.conv_pool = nn.Sequential(*(
            list(alexnet.features.children())[:layers_to_stack] +
            [
                Flatten(num_params), 
                nn.Dropout(p=dropout_ratio), 
                nn.Linear(num_params, last_layer_num_params)
            ]
        ))

    def forward(self, x):
        return self.conv_pool(x)


class HDF5Dataset(torch.utils.data.Dataset):
    """A dataset wrapping a loaded HDF5 file."""

    def __init__(self, hdf5):
        assert 'images' in hdf5 and 'labels' in hdf5, \
            "The HDF5 file must provide both 'images' and 'labels'."
        assert hdf5['images'].shape[0] == hdf5['labels'].shape[0], \
            "Images and labels must have the same number of elements."

        self._len = hdf5['images'].shape[0]

        self.hdf5 = hdf5

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self.hdf5['images'][index], self.hdf5['labels'][index]
