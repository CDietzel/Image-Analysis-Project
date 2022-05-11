from cgitb import grey
from matplotlib.pyplot import axis, close, hot
import numpy as np
from numpy.random import default_rng
import random
import copy
import math


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
        hist = hist.astype(np.uint)
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
            result = np.sum(filter * gray_img[i : i + f_w, j : j + f_h])
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
            dir = np.mod(dir, np.pi * 2)  # normalize to 0-2pi scale
        elif method == "sobel":
            filter_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            filter_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
            scale = 8
            dx = self.linear_filter(gray_img, filter_x, scale=scale, dtype=np.int16)
            dy = self.linear_filter(gray_img, filter_y, scale=scale, dtype=np.int16)
            mag = np.sqrt((dx * dx) + (dy * dy))
            dir = np.arctan2(dy, dx)
            dir = np.mod(dir, np.pi * 2)  # normalize to 0-2pi scale
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
            dir = np.argmax(dstack, axis=0)
            dir = (dir * np.pi) / 4  # normalize to 0-2pi scale
        dir = (dir * 255) / (np.pi * 2)  # re-normalize to 0-255 scale for easy viewing
        mag = mag * 10  # scaling factor so that the magnitude image is actually visible
        mag = np.rint(mag)
        mag = mag.astype(np.uint8)
        dir = np.rint(dir)
        dir = dir.astype(np.uint8)
        return mag, dir

    def _translate(self, bin_img, trans_x, trans_y):
        if trans_x > 0:
            bin_img = np.pad(bin_img, ((0, 0), (trans_x, 0)), mode="constant")[
                :, :-trans_x
            ]
        elif trans_x < 0:
            trans_x = -trans_x
            bin_img = np.pad(bin_img, ((0, 0), (0, trans_x)), mode="constant")[
                :, trans_x:
            ]
        if trans_y > 0:
            bin_img = np.pad(bin_img, ((trans_y, 0), (0, 0)), mode="constant")[
                :-trans_y, :
            ]
        elif trans_y < 0:
            trans_y = -trans_y
            bin_img = np.pad(bin_img, ((0, trans_y), (0, 0)), mode="constant")[
                trans_y:, :
            ]
        return bin_img

    def dilation(self, bin_img, strel, hot_x, hot_y):
        strel = np.array(strel)
        dil_img = np.zeros(bin_img.shape)
        for i, j in np.ndindex(strel.shape):
            if strel[i, j] > 0:
                trans_x = j - hot_x
                trans_y = i - hot_y
                trans_img = self._translate(bin_img, trans_x, trans_y)
                dil_img = np.logical_or(trans_img > 0, dil_img > 0)
        dil_img = dil_img * 255
        dil_img = dil_img.astype(np.uint8)
        return dil_img

    def erosion(self, bin_img, strel, hot_x, hot_y):
        strel = np.array(strel)
        y, x = strel.shape
        pad_l = hot_x
        pad_r = x - hot_x - 1
        pad_u = hot_y
        pad_d = y - hot_y - 1
        ero_img = np.zeros(bin_img.shape)
        pad_img = np.pad(bin_img, ((pad_u, pad_d), (pad_l, pad_r)), mode="constant")
        for i, j in np.ndindex(bin_img.shape):
            logical_pad = pad_img[i : i + y, j : j + x] > 0
            logical_strel = strel > 0
            and_pad_strel = np.logical_and(logical_pad, logical_strel)
            if np.array_equal(and_pad_strel, logical_strel):
                ero_img[i, j] = 255
        ero_img = ero_img.astype(np.uint8)
        return ero_img

    def histogram_thresh(self, gray_img):
        limit = 256
        var_w = []
        prob = []
        prior_o = []
        prior_b = []
        mean_o = []
        mean_b = []
        var_o = []
        var_b = []
        wgv = []
        hist = self.calc_histogram(gray_img)
        for i in range(limit):
            p = hist[i] / np.sum(hist)
            prob.append(p)

        for T in range(limit):
            prior = 0
            for i in range(0, T + 1):
                prior += prob[i]
            prior_o.append(prior)

        for T in range(limit):
            prior = 0
            for i in range(T + 1, 256):
                prior += prob[i]
            prior_b.append(prior)

        for T in range(limit):
            mean = 0
            for i in range(0, T + 1):
                mean += i * prob[i]
            divisor = prior_o[T]
            if divisor == 0:
                result = 0
            else:
                result = mean / divisor
            if math.isnan(result):
                result = 0
            mean_o.append(result)

        for T in range(limit):
            mean = 0
            for i in range(T + 1, 256):
                mean += i * prob[i]
            divisor = prior_b[T]
            if divisor == 0:
                result = 0
            else:
                result = mean / divisor
            if math.isnan(result):
                result = 0
            mean_b.append(result)

        for T in range(limit):
            var = 0
            for i in range(0, T + 1):
                var += ((i - mean_o[T]) ** 2) * prob[i]
            divisor = prior_o[T]
            if divisor == 0:
                result = 0
            else:
                result = var / divisor
            if math.isnan(result):
                result = 0
            var_o.append(result)

        for T in range(limit):
            var = 0
            for i in range(T + 1, 256):
                var += ((i - mean_b[T]) ** 2) * prob[i]
            divisor = prior_b[T]
            if divisor == 0:
                result = 0
            else:
                result = var / divisor
            if math.isnan(result):
                result = 0
            var_b.append(result)

        for T in range(limit):
            var_o_t = var_o[T]
            prior_o_t = prior_o[T]
            var_b_t = var_b[T]
            prior_b_t = prior_b[T]
            var_w = var_o_t * prior_o_t + var_b_t * prior_b_t
            wgv.append(var_w)

        wgv = np.array(wgv)
        T = np.argmin(wgv)
        thresh_img = gray_img < T
        thresh_img = thresh_img * 255
        thresh_img = thresh_img.astype(np.uint8)
        return thresh_img

    def k_means_clustering(self, gray_img, k, use_loc=True, init_clust=None):
        h, w = gray_img.shape
        cluster_img = np.zeros(gray_img.shape)
        clusters = []
        metric_arr = np.expand_dims(gray_img, axis=-1)
        if use_loc:
            pos_grid = np.transpose(np.meshgrid(np.arange(h), np.arange(w)))
            metric_arr = np.dstack((metric_arr, pos_grid))
        metric_arr = metric_arr.reshape(h * w, -1)
        metric_arr = np.unique(metric_arr, axis=0)
        if init_clust is None:
            clusters = self._rng.choice(metric_arr, k, replace=False, axis=0)
        else:
            clusters = np.array(init_clust)

        not_done = True
        while not_done:
            old_cluster_img = copy.deepcopy(cluster_img)
            metric_arr = np.expand_dims(gray_img, axis=-1).astype(int)
            if use_loc:
                pos_grid = np.transpose(np.meshgrid(np.arange(h), np.arange(w)))
                metric_arr = np.dstack((metric_arr, pos_grid))
            dist = []
            for cluster in clusters:
                diff = metric_arr - cluster
                if use_loc:
                    diff = diff.astype(float)
                    diff[:, :, 1:3] *= 0.05  # scaling factor so that the pixel position
                    # error doesn't overwhelm the pixel value error
                cluster_diff = np.linalg.norm(diff, axis=-1)
                dist.append(cluster_diff)
            dist = np.array(dist)
            cluster_img = np.argmin(dist, axis=0)
            cluster_avg = []
            for num in range(len(clusters)):
                mask = cluster_img == num
                masked_metric = metric_arr[mask]
                mean_metric = masked_metric.mean(axis=0)
                cluster_avg.append(mean_metric)
            cluster_avg = np.array(cluster_avg)
            cluster_avg[np.isnan(cluster_avg)] = clusters[np.isnan(cluster_avg)]
            clusters = cluster_avg
            not_done = not np.array_equal(old_cluster_img, cluster_img)
        cluster_img = cluster_img * 255
        cluster_img = cluster_img / (k - 1)
        cluster_img = cluster_img.astype(np.uint8)
        return cluster_img

    def extract_features(self, gray_img):
        seg_img = self.k_means_clustering(
            gray_img, 2, use_loc=False, init_clust=[[255], [0]]
        )
        # compute area of cell:
        area = np.count_nonzero(seg_img)
        # compute perimeter of cell:
        dil_seg_img = self.dilation(
            seg_img, strel=[[0, 1, 0], [1, 1, 1], [0, 1, 0]], hot_x=1, hot_y=1
        )
        int_bound = np.logical_xor(seg_img, dil_seg_img)
        perimeter = np.count_nonzero(int_bound)
        # compute median cell pixel magnitude:
        mask = seg_img == 0
        mask_img = np.ma.array(gray_img, mask=mask)
        median = np.ma.median(mask_img)
        # compute standard deviation of cell pixel magnitudes:
        std = np.ma.std(mask_img)
        feature_vector = [area, perimeter, median, std]
        return feature_vector


# if __name__ == "__main__":
#     q = ImageManipulator()
#     # img = np.array(
#     #     [[7, 15, 21, 13], [31, 22, 25, 23], [13, 18, 10, 16], [25, 24, 29, 18]]
#     # )
#     img = np.array(
#         [[5, 15, 5, 15], [15, 5, 15, 5], [5, 15, 5, 15], [35, 25, 35, 25]]
#     )
#     feature_vector = q.extract_features(img)
#     pass
