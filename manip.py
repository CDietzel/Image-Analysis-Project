from cgitb import grey
import random
from pathlib import Path

from numpy.random import default_rng


import numpy as np
import yaml
from matplotlib import pyplot as plt
from skimage import io


class ImageManipulator:
    def __init__(self):
        self._rng = default_rng()

    def to_greyscale(self, rgb_img, channel=None):
        grey_img = rgb_img
        if channel is None:
            grey_img = np.mean(grey_img, axis=2)
            grey_img = np.rint(grey_img)
        elif channel == "R":
            grey_img = grey_img[:, :, 0]
        elif channel == "G":
            grey_img = grey_img[:, :, 1]
        elif channel == "B":
            grey_img = grey_img[:, :, 2]
        else:
            raise ValueError('channel must be "R", "G", "B", or None')
        grey_img = grey_img.astype(int)
        return grey_img

    def salt_pepper_noise(self, grey_img, ratio):
        img = grey_img
        mask = np.random.choice(
            [-1, 0, 255], size=grey_img.shape, p=[1 - ratio, ratio / 2, ratio / 2]
        )
        np.copyto(mask, img, where=mask == -1)
        img = mask.astype(int)
        return img


if __name__ == "__main__":
    random.seed(42)
    m = ImageManipulator()
    img = io.imread("./Cancerous cell smears/cyl01.BMP")
    img = m.to_greyscale(img)
    img = m.salt_pepper_noise(img, 0.1)
    plt.imshow(img, vmin=0, vmax=255)
    plt.show()
