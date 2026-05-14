"""
Miscellaneous utilities for the depth mapping project.

This module contains helper classes and functions used throughout
the training and evaluation pipeline.
"""

class AverageMeter(object):
    """
    Computes and stores the average and current value.

    Useful for tracking metrics during training such as loss values.
    """

    def __init__(self):
        """Initialize the AverageMeter."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset all stored values to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update the meter with a new value.

        Args:
            val (float): New value to add.
            n (int): Number of occurrences of this value (for weighted average).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count