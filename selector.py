"""
Model selector for dynamic model loading.

This module enables loading model classes by string name,
providing a convenient interface for model selection.
"""

import models


# TODO: just an example
class Autoencoder(models.Autoencoder):
    """
    Autoencoder model wrapper for depth mapping.

    Pre-configured Autoencoder with 3 input channels for RGB images.
    """

    def __init__(self):
        """Initialize the Autoencoder model."""
        super(Autoencoder, self).__init__(input_channels=3)

class UNet(models.UNet):
    """
    U-Net model wrapper for depth mapping.

    Pre-configured U-Net with 3 input channels and 1 output channel for depth.
    """

    def __init__(self):
        """Initialize the U-Net model."""
        super(UNet, self).__init__(n_channels=3, n_classes=1)


