import numpy as np
from numpy.random import default_rng


class ImageManipulator:
    def __init__(self):
        self._rng = default_rng(seed=42)

    def salt_pepper_noise(self, gray_img, ratio):
        noise = self._rng.choice(
            [-1, 0, 255], size=gray_img.shape, p=[1 - ratio, ratio / 2, ratio / 2]
        )
        np.copyto(noise, gray_img, where=noise == -1)
        gray_img = noise.astype(np.uint8)
        return gray_img

    def gaussian_noise(self, gray_img, mean, std):
        noise = self._rng.normal(loc=mean, scale=std, size=gray_img.shape)
        gray_img = gray_img + noise
        gray_img = np.clip(gray_img, 0, 255)
        gray_img = np.rint(gray_img)
        gray_img = gray_img.astype(np.uint8)
        return gray_img

    def calc_histogram(self, gray_img):
        hist = np.zeros(256)
        for i in range(len(hist)):
            hist[i] = np.sum(gray_img == i)
        hist = hist.astype(np.uint8)
        return hist

    def avg_histograms(self, hist_list):
        hist_arr = np.array(hist_list)
        hist = np.mean(hist_arr, axis=0)
        hist = np.rint(hist)
        hist = hist.astype(np.uint8)
        return hist

    def hist_equalization(self, gray_img, hist=None):
        if hist is None:
            hist = self.calc_histogram(gray_img)
        bins = range(len(hist))
        hist = hist / np.sum(hist)
        cs = np.cumsum(hist)
        cs = (len(hist) - 1) * cs / cs[-1]
        gray_img = np.interp(gray_img, bins, cs)
        gray_img = np.rint(gray_img)
        gray_img = gray_img.astype(np.uint8)
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
        gray_img = gray_img.astype(np.uint8)
        return gray_img

    def linear_filter(self, gray_img, filter, scale, dtype=np.uint8):
        filter = np.array(filter)
        f_w, f_h = filter.shape
        i_w, i_h = gray_img.shape
        o_w = i_w - f_w + 1
        o_h = i_h - f_h + 1
        new_img = np.zeros((o_w, o_h))
        for i, j in np.ndindex(new_img.shape):
            neighborhood = gray_img[i : i + f_w, j : j + f_h]
            result = np.sum(filter * neighborhood)
            scaled_result = result / scale
            new_img[i, j] = scaled_result
        new_img = np.rint(new_img)
        new_img = new_img.astype(dtype)
        return new_img

    def median_filter(self, gray_img, weights):
        weights = np.array(weights)
        f_w, f_h = weights.shape
        i_w, i_h = gray_img.shape
        o_w = i_w - f_w + 1
        o_h = i_h - f_h + 1
        new_img = np.zeros((o_w, o_h))
        for i, j in np.ndindex(new_img.shape):
            pixel_list = np.array([])
            for k, l in np.ndindex(f_w, f_h):
                pixel_list = np.append(
                    pixel_list,
                    [gray_img[i : i + f_w, j : j + f_h][k, l]] * weights[k, l],
                )
            result = np.median(pixel_list)
            new_img[i, j] = result
        new_img = np.rint(new_img)
        new_img = new_img.astype(np.uint8)
        return new_img

    def edge_detect(self, gray_img, method="prewitt"):
        if method == "prewitt":
            filter_x = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
            filter_y = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
            scale = 6
            dx = self.linear_filter(gray_img, filter_x, scale=scale, dtype=np.int16)
            dy = self.linear_filter(gray_img, filter_y, scale=scale, dtype=np.int16)
            mag = np.sqrt((dx * dx) + (dy * dy))
            dir = np.arctan2(dy, dx)
        elif method == "sobel":
            filter_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            filter_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
            scale = 8
            dx = self.linear_filter(gray_img, filter_x, scale=scale, dtype=np.int16)
            dy = self.linear_filter(gray_img, filter_y, scale=scale, dtype=np.int16)
            mag = np.sqrt((dx * dx) + (dy * dy))
            dir = np.arctan2(dy, dx)
        elif method == "compass":
            filter_0 = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            filter_1 = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]
            filter_2 = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
            filter_3 = [[0, -1, -2], [1, 0, -1], [2, 1, 0]]
            scale = 8
            d0 = self.linear_filter(gray_img, filter_0, scale=scale, dtype=np.int16)
            d1 = self.linear_filter(gray_img, filter_1, scale=scale, dtype=np.int16)
            d2 = self.linear_filter(gray_img, filter_2, scale=scale, dtype=np.int16)
            d3 = self.linear_filter(gray_img, filter_3, scale=scale, dtype=np.int16)
            d4 = -d0
            d5 = -d1
            d6 = -d2
            d7 = -d3
            dstack = np.array([d0, d1, d2, d3, d4, d5, d6, d7])
            mag = np.amax(dstack, axis=0)
            dir = np.argmax(dstack, axis=0)  # NOTE: NEED TO FIX THIS SCALING
        return mag, dir  # NOTE: Maybe add datatype conversion to int? IDK.


if __name__ == "__main__":
    q = ImageManipulator()
    img = np.array(
        [[7, 15, 21, 13], [31, 22, 25, 23], [13, 18, 10, 16], [25, 24, 29, 18]]
    )
    mag, dir = q.edge_detect(img, method="compass")
    pass
