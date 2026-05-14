import os
import torch
import csv
from torch.utils.data import Dataset
import numpy as np
import imageio
import augmentations as aug

# img: 0-255
# depth
SMALL_EPS = 1e-6

"""
NYU dataset loaders for monocular depth mapping.

This module provides PyTorch Dataset classes for loading and preprocessing
the NYU Depth dataset for training and testing depth mapping models.
"""

class NYUTrainset(Dataset):
    """
    PyTorch Dataset for NYU Depth training data.

    Loads RGB images and corresponding depth maps from the NYU dataset,
    applies preprocessing and optional data augmentations.
    """

    def __init__(self, index_file, aug=None, debug=False):
        """
        Initialize the NYU training dataset.

        Args:
            index_file (str): Path to CSV file containing image and depth map paths.
            aug (aug.Compose, optional): Data augmentation pipeline to apply.
            debug (bool): If True, saves debug images and returns early.
        """
        self.DEBUG = debug
        self.augmentation = aug

        with open(index_file, 'r') as f:
            self.data_path = list(csv.reader(f))

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of data samples.
        """
        return len(self.data_path)

    def __getitem__(self, index):
        """
        Get a sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: Preprocessed image and depth map tensors.
        """
        img = np.asarray(imageio.imread(self.data_path[index][0]), dtype=np.float)
        depth = np.asarray(imageio.imread(self.data_path[index][1]), dtype=np.float)
        depth_min = np.min(depth)
        depth_max = np.max(depth)

        if self.DEBUG:
            import cv2
            cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'depth.png'), cv2.cvtColor(depth, cv2.COLOR_RGB2BGR))
            return 1

        img = np.transpose(img, [2, 0, 1]) / 255
        depth = (depth - depth_min) / (depth_max - depth_min + SMALL_EPS)
        depth = depth[np.newaxis, ...]
        sample = (img, depth)
        if self.augmentation is not None:
            sample = self.augmentation(sample)

        return sample


class NYUTestset(Dataset):
    """
    PyTorch Dataset for NYU Depth testing data.

    Loads RGB images and corresponding depth maps from the NYU dataset
    for evaluation purposes, without data augmentations.
    """

    def __init__(self, index_file):
        """
        Initialize the NYU test dataset.

        Args:
            index_file (str): Path to CSV file containing image and depth map paths.
        """
        with open(index_file, 'r') as f:
            self.data_path = list(csv.reader(f))

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of data samples.
        """
        return len(self.data_path)

    def __getitem__(self, index):
        """
        Get a sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: Preprocessed image and depth map tensors.
        """
        img = np.asarray(imageio.imread(self.data_path[index][0]), dtype=np.float)
        depth = np.asarray(imageio.imread(self.data_path[index][1]), dtype=np.float)
        depth_min = np.min(depth)
        depth_max = np.max(depth)

        img = np.transpose(img, [2, 0, 1]) / 255
        depth = (depth - depth_min) / (depth_max - depth_min + SMALL_EPS)
        depth = depth[np.newaxis, ...]
        sample = (img, depth)

        return sample


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils import data
    bs = 1
    augs = aug.Compose([aug.RandomHorizontalFlip()])
    dst = NYUTrainset(index_file='data/train_debug.csv', aug=augs, debug=False)
    dst_test = NYUTestset(index_file='data/train_debug.csv')
    trainloader = data.DataLoader(dst_test, batch_size=1)
    for i, data in enumerate(trainloader):
        img, depth = data
        print(img)
        print(depth)
        img = np.transpose(img, [0, 2, 3, 1])
        f, axarr = plt.subplots(2)
        for j in range(bs):
            axarr[0].imshow(img[j])
            axarr[1].imshow(depth[j][0])
        plt.show()
        a = input()
        if a == 'ex':
            break
        else:
            plt.close()