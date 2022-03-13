import random
from statistics import median, median_grouped
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
        if channel is None:
            gray_img = np.mean(rgb_img, axis=2)
            gray_img = np.rint(gray_img)
        elif channel == "R":
            gray_img = rgb_img[:, :, 0]
        elif channel == "G":
            gray_img = rgb_img[:, :, 1]
        elif channel == "B":
            gray_img = rgb_img[:, :, 2]
        else:
            raise ValueError('channel must be "R", "G", "B", or None')
        gray_img = gray_img.astype(int)
        return gray_img

    def salt_pepper_noise(self, gray_img, ratio):
        noise = self._rng.choice(
            [-1, 0, 255], size=gray_img.shape, p=[1 - ratio, ratio / 2, ratio / 2]
        )
        np.copyto(noise, gray_img, where=noise == -1)
        gray_img = noise.astype(int)
        return gray_img

    def gaussian_noise(self, gray_img, mean, std):
        noise = self._rng.normal(loc=mean, scale=std, size=gray_img.shape)
        gray_img = gray_img + noise
        gray_img = np.clip(gray_img, 0, 255)
        gray_img = np.rint(gray_img)
        gray_img = gray_img.astype(int)
        return gray_img

    def calc_histogram(self, gray_img):
        hist = np.zeros(256)
        for i in range(len(hist)):
            hist[i] = np.sum(gray_img == i)
        hist = hist.astype(int)
        return hist

    def avg_histograms(self, hist_list):
        hist_arr = np.array(hist_list)
        hist = np.mean(hist_arr, axis=0)
        hist = np.rint(hist)
        hist = hist.astype(int)
        return hist

    def hist_equalization(self, gray_img, hist):
        bins = range(len(hist))
        hist = hist / np.sum(hist)
        cs = np.cumsum(hist)
        cs = (len(hist) - 1) * cs / cs[-1]
        gray_img = np.interp(gray_img, bins, cs)
        gray_img = np.rint(gray_img)
        gray_img = gray_img.astype(int)
        return gray_img

    def quantize_image(self, gray_img, thresholds):
        t = np.array(thresholds)
        if t[0] != 0:
            t = np.insert(t, 0, 0)
        if t[-1] != 256:
            t = np.append(t, 256)
        P = len(t) - 1
        r = np.zeros(P)
        for i in range(len(r)):
            r[i] = (t[i] + t[i + 1]) / 2
        Q = np.zeros(256)
        x = np.array(range(256))
        for i in range(P):
            Q = Q + r[i] * ((x >= t[i]) & (x < t[i + 1]))
        B = Q[gray_img]
        gray_img = np.rint(B)
        gray_img = gray_img.astype(int)
        return gray_img

    def linear_filter(self, gray_img, filter, scale):
        f_w, f_h = filter.shape
        i_w, i_h = gray_img.shape
        o_w = i_w - f_w + 1
        o_h = i_h - f_h + 1
        new_img = np.zeros((o_w, o_h))
        for i in range(o_w):
            for j in range(o_h):
                neighborhood = gray_img[i : i + f_w, j : j + f_h]
                result = np.sum(filter * neighborhood)
                scaled_result = result / scale
                new_img[i, j] = scaled_result
        new_img = np.rint(new_img)
        new_img = new_img.astype(int)
        return new_img

    def median_filter(self, gray_img, weights):
        f_w, f_h = weights.shape
        i_w, i_h = gray_img.shape
        o_w = i_w - f_w + 1
        o_h = i_h - f_h + 1
        new_img = np.zeros((o_w, o_h))
        for i, j in np.ndindex(new_img.shape):
            neighborhood = gray_img[i : i + f_w, j : j + f_h]
            pixel_list = np.array([])
            for k, l in np.ndindex(neighborhood.shape):
                pixel_list = np.append(pixel_list, [neighborhood[k, l]] * weights[k, l])
            result = np.median(pixel_list)
            new_img[i, j] = result
        new_img = np.rint(new_img)
        new_img = new_img.astype(int)
        return new_img

    def mean_square_quantization_error(self, gray_img, quant_img):
        error = quant_img - gray_img
        square_error = np.square(error)
        mean_square_error = np.mean(square_error)
        return mean_square_error


if __name__ == "__main__":
    random.seed(42)
    m = ImageManipulator()
    img = io.imread("./Cancerous cell smears/cyl01.BMP")
    img = m.to_grayscale(img)
    # img = m.salt_pepper_noise(img, 0.01)
    # img = m.gaussian_noise(img, 0, 10)
    hist = m.calc_histogram(img)
    img = m.hist_equalization(img, hist)
    img2 = m.quantize_image(img, [7, 10, 100, 200, 213])
    # img = m.linear_filter(img, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), 9)
    # img = m.median_filter(img, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    print(m.mean_square_quantization_error(img, img2))
    plt.imshow(img)
    plt.show()
