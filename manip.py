import random
from pathlib import Path

import numpy as np
import yaml
from matplotlib import pyplot as plt
from numpy.random import default_rng
from skimage import io


class ImageManipulator:
    def __init__(self):
        self._rng = default_rng(seed=42)

    def to_grayscale(self, rgb_img, channel=None):
        gray_img = rgb_img
        if channel is None:
            gray_img = np.mean(gray_img, axis=2)
            gray_img = np.rint(gray_img)
        elif channel == "R":
            gray_img = gray_img[:, :, 0]
        elif channel == "G":
            gray_img = gray_img[:, :, 1]
        elif channel == "B":
            gray_img = gray_img[:, :, 2]
        else:
            raise ValueError('channel must be "R", "G", "B", or None')
        gray_img = gray_img.astype(int)
        return gray_img

    def salt_pepper_noise(self, gray_img, ratio):
        img = gray_img
        noise = self._rng.choice(
            [-1, 0, 255], size=gray_img.shape, p=[1 - ratio, ratio / 2, ratio / 2]
        )
        np.copyto(noise, img, where=noise == -1)
        img = noise.astype(int)
        return img

    def gaussian_noise(self, gray_img, mean, std):
        img = gray_img
        noise = self._rng.normal(loc=mean, scale=std, size=gray_img.shape)
        img = img + noise
        img = np.clip(img, 0, 255)
        img = np.rint(img)
        img = img.astype(int)
        return img

    def calc_histogram(self, gray_img):
        img = gray_img
        hist = np.zeros(256)
        for i in range(len(hist)):
            hist[i] = np.sum(img == i)
        hist = hist.astype(int)
        return hist

    def avg_histograms(self, hist_list):
        hist_arr = np.array(hist_list)
        hist = np.mean(hist_arr, axis=0)
        hist = np.rint(hist)
        hist = hist.astype(int)
        return hist

    def hist_equalization(self, gray_img, hist):
        img = gray_img
        bins = range(len(hist))
        hist = hist / np.sum(hist)
        cs = np.cumsum(hist)
        cs = (len(hist) - 1) * cs / cs[-1]
        img = np.interp(img, bins, cs)
        img = np.rint(img)
        img = img.astype(int)
        return img

    def quantize_image(self, gray_img, num_levels):
        img = gray_img
        ratio = num_levels / 256
        img = img * ratio
        img = img.astype(int)
        return img


if __name__ == "__main__":
    random.seed(42)
    m = ImageManipulator()
    img = io.imread("./Cancerous cell smears/cyl01.BMP")
    img = m.to_grayscale(img)
    # img = m.salt_pepper_noise(img, 0.01)
    # img = m.gaussian_noise(img, 0, 10)
    hist = m.calc_histogram(img)
    img = m.hist_equalization(img, hist)
    img = m.quantize_image(img, 3)
    plt.imshow(img)
    plt.show()
