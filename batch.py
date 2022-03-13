import random
from pathlib import Path
from statistics import median, median_grouped

import numpy as np
import yaml
from matplotlib import pyplot as plt
from numpy.random import default_rng
from skimage import io
from manip import ImageManipulator


class BatchProcessor:
    def __init__(self, config_file=None):
        if config_file is not None:
            with open(config_file, "r") as file:
                self._config = yaml.safe_load(file)
        self._image_sets = {}
        self._statistics = {}

    def set_config(self, config_file):
        with open(config_file, "r") as file:
            self._config = yaml.safe_load(file)

    def process(self):
        pass

    def load_image_set(self, arg_batch_name, file_prefix=None, channel=None):
        # load in all image files beginning with file_prefix in self._config["input_image_path"]
        # convert them to greyscale using channel
        # store the list in self._image_sets with key arg_batch_name
        pass

    def save_image_set(self, arg_batch_name, file_prefix=None):
        # save all images that are in self._image_sets with key arg_batch_name
        # store the files in self._config["output_image_path"]
        # the filename of each is file_prefix+number.BMP (the numbers should increment)
        pass

    def save_statistics(self, arg_batch_name, file_prefix=None):
        # save all statistics that are in self._statistics with key arg_batch_name
        # store the files in self._config["output_statistics_path"]
        # the filename of each is file_prefix+number.yaml (the numbers should increment)
        pass

    def _to_grayscale(self, rgb_img, channel):
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

    def _mean_square_quantization_error(self, gray_img, quant_img):
        error = quant_img - gray_img
        square_error = np.square(error)
        mean_square_error = np.mean(square_error)
        return mean_square_error
