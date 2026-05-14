import math
import numbers
import random
import numpy as np

from PIL import Image, ImageOps

"""
Data augmentation utilities for monocular depth mapping.

This module provides classes for composing and applying various data augmentations
to images and depth maps used in the depth mapping training pipeline.
"""

class Compose(object):
    """
    Composes several augmentations together.

    This class allows chaining multiple augmentation transformations
    that will be applied sequentially to the input data.
    """

    def __init__(self, augmentations):
        """
        Initialize the Compose object.

        Args:
            augmentations (list): List of augmentation objects to compose.
        """
        self.augmentations = augmentations

    def __call__(self, items):
        """
        Apply the composed augmentations to the input items.

        Args:
            items (tuple): Tuple of items (e.g., image and depth map) to augment.

        Returns:
            tuple: Augmented items.
        """
        #print('{} {}'.format(img.shape, label.shape))
        #img, label = Image.fromarray(img, mode='F'), Image.fromarray(label, mode='F')   
        items = list(items)         
        for a in self.augmentations:
            items = a(items)
        return tuple(items)

class RandomHorizontalFlip(object):
    """
    Randomly flips the input items horizontally with 50% probability.

    This augmentation helps the model become invariant to horizontal orientation.
    """

    def __call__(self, items):
        """
        Apply random horizontal flip to the items.

        Args:
            items (list): List of numpy arrays to flip.

        Returns:
            list: Flipped items if random condition met, otherwise original items.
        """
        if random.random() < 0.5:
            items = [np.fliplr(item).copy() for item in items]
        return items