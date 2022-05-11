import os
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from skimage import io

from manip import ImageManipulator


def main():
    config_file_path = "./config.yaml"
    b = BatchProcessor(config_file=config_file_path)
    b.process()


class BatchProcessor:
    def __init__(self, config_file=None):
        if config_file is not None:
            with open(config_file, "r") as file:
                self._config = yaml.safe_load(file)
        self._image_sets = {}
        self._statistics = []
        self._datasets = {}
        self._m = ImageManipulator()
        save_path = self._config["output_image_path"]
        Path(save_path).mkdir(parents=True, exist_ok=True)
        for f in os.listdir(save_path):
            os.remove(os.path.join(save_path, f))
        stats_path = self._config["output_statistics_path"]
        Path(stats_path).mkdir(parents=True, exist_ok=True)
        for f in os.listdir(stats_path):
            os.remove(os.path.join(stats_path, f))
        datasets_path = self._config["output_datasets_path"]
        Path(datasets_path).mkdir(parents=True, exist_ok=True)
        # Don't always delete the datasets for no reason
        # for f in os.listdir(datasets_path):
        #     os.remove(os.path.join(datasets_path, f))
        hist_path = self._config["histogram_path"]
        Path(hist_path).mkdir(parents=True, exist_ok=True)
        for f in os.listdir(hist_path):
            os.remove(os.path.join(hist_path, f))

    def set_config(self, config_file):
        with open(config_file, "r") as file:
            self._config = yaml.safe_load(file)

    def process(self):
        steps = self._config["processing_steps"]
        for step in steps:
            batch_start_time = time.time()
            function_name = step["function"]

            if function_name == "load_image_set":
                in_path = self._config["input_image_path"]
                prefix = step["file_prefix"]
                channel = step["channel"]
                return_name = step["return_batch_name"]
                file_path_list = []
                for file in os.listdir(in_path):
                    if file.startswith(prefix) and file != "super45.BMP":
                        # super45.BMP is a corrupted image file,
                        # so it doesn't load properly.
                        # Therefore, skip it.
                        file_path_list.append(os.path.join(in_path, file))
                image_list = []
                for path in file_path_list:
                    img = io.imread(path)
                    img = self._to_grayscale(img, channel)
                    image_list.append(img)
                self._image_sets[return_name] = image_list

            elif function_name == "save_image_set":
                save_path = self._config["output_image_path"]
                prefix = step["file_prefix"]
                batch_name = step["arg_batch_name"]
                image_list = self._image_sets[batch_name]
                for i, image in enumerate(image_list):
                    out_path = os.path.join(save_path, prefix + str(i) + ".BMP")
                    io.imsave(out_path, image)

            elif function_name == "save_statistics":
                out_path = self._config["output_statistics_path"]
                prefix = step["file_prefix"]
                out_path = os.path.join(out_path, prefix + ".yaml")
                with open(out_path, "w") as file:
                    yaml.dump(self._statistics, file, default_flow_style=None)

            elif function_name == "salt_pepper_noise":
                batch_name = step["arg_batch_name"]
                return_name = step["return_batch_name"]
                ratio = step["ratio"]
                image_list = self._image_sets[batch_name]
                new_image_list = []
                runtime_list = []
                for image in image_list:
                    image_start_time = time.time()
                    new_image_list.append(self._m.salt_pepper_noise(image, ratio))
                    image_elapsed = time.time() - image_start_time
                    runtime_list.append(image_elapsed)
                self._image_sets[return_name] = new_image_list
                batch_elapsed = time.time() - batch_start_time
                avg_runtime = statistics.mean(runtime_list)
                stats = {
                    "entire_batch_runtime": batch_elapsed,
                    "avg_image_runtime": avg_runtime,
                }
                stats_data = [function_name, stats]
                self._statistics.append(stats_data)

            elif function_name == "gaussian_noise":
                batch_name = step["arg_batch_name"]
                return_name = step["return_batch_name"]
                mean = step["mean"]
                std = step["std"]
                image_list = self._image_sets[batch_name]
                new_image_list = []
                runtime_list = []
                for image in image_list:
                    image_start_time = time.time()
                    new_image_list.append(self._m.gaussian_noise(image, mean, std))
                    image_elapsed = time.time() - image_start_time
                    runtime_list.append(image_elapsed)
                self._image_sets[return_name] = new_image_list
                batch_elapsed = time.time() - batch_start_time
                avg_runtime = statistics.mean(runtime_list)
                stats = {
                    "entire_batch_runtime": batch_elapsed,
                    "avg_image_runtime": avg_runtime,
                }
                stats_data = [function_name, stats]
                self._statistics.append(stats_data)

            elif function_name == "calc_histogram":
                batch_name = step["arg_batch_name"]
                hist_path = self._config["histogram_path"]
                prefix = step["file_prefix"]
                image_list = self._image_sets[batch_name]
                new_hist_list = []
                runtime_list = []
                for image in image_list:
                    image_start_time = time.time()
                    new_hist_list.append(self._m.calc_histogram(image).tolist())
                    image_elapsed = time.time() - image_start_time
                    runtime_list.append(image_elapsed)
                new_avg_hist = self._m.avg_histograms(new_hist_list).tolist()
                batch_elapsed = time.time() - batch_start_time
                avg_runtime = statistics.mean(runtime_list)
                stats = {
                    "entire_batch_runtime": batch_elapsed,
                    "avg_image_runtime": avg_runtime,
                }
                stats_data = [function_name, stats]
                self._statistics.append(stats_data)
                out_path = os.path.join(hist_path, prefix + str(0) + ".png")
                plt.bar(range(256), new_hist_list[0], width=1)
                plt.savefig(out_path)
                plt.clf()
                out_path = os.path.join(hist_path, "avg_" + prefix + ".png")
                plt.bar(range(256), new_avg_hist, width=1)
                plt.savefig(out_path)
                plt.clf()

            elif function_name == "hist_equalization":
                batch_name = step["arg_batch_name"]
                return_name = step["return_batch_name"]
                image_list = self._image_sets[batch_name]
                new_image_list = []
                runtime_list = []
                for image in image_list:
                    image_start_time = time.time()
                    new_image_list.append(self._m.hist_equalization(image))
                    image_elapsed = time.time() - image_start_time
                    runtime_list.append(image_elapsed)
                self._image_sets[return_name] = new_image_list
                batch_elapsed = time.time() - batch_start_time
                avg_runtime = statistics.mean(runtime_list)
                stats = {
                    "entire_batch_runtime": batch_elapsed,
                    "avg_image_runtime": avg_runtime,
                }
                stats_data = [function_name, stats]
                self._statistics.append(stats_data)

            elif function_name == "quantize_image":
                batch_name = step["arg_batch_name"]
                return_name = step["return_batch_name"]
                thresholds = step["thresholds"]
                image_list = self._image_sets[batch_name]
                new_image_list = []
                runtime_list = []
                msqe_list = []
                for image in image_list:
                    image_start_time = time.time()
                    quant_img = self._m.quantize_image(image, thresholds)
                    image_elapsed = time.time() - image_start_time
                    new_image_list.append(quant_img)
                    msqe_list.append(
                        self._mean_square_quantization_error(image, quant_img)
                    )
                    runtime_list.append(image_elapsed)
                self._image_sets[return_name] = new_image_list
                batch_elapsed = time.time() - batch_start_time
                avg_runtime = statistics.mean(runtime_list)
                avg_msqe = float(statistics.mean(msqe_list))
                stats = {
                    "entire_batch_runtime": batch_elapsed,
                    "avg_image_runtime": avg_runtime,
                    "avg_msqe": avg_msqe,
                }
                stats_data = [function_name, stats]
                self._statistics.append(stats_data)

            elif function_name == "linear_filter":
                batch_name = step["arg_batch_name"]
                return_name = step["return_batch_name"]
                filter = step["filter"]
                scale = step["scale"]
                image_list = self._image_sets[batch_name]
                new_image_list = []
                runtime_list = []
                for image in image_list:
                    image_start_time = time.time()
                    new_image_list.append(self._m.linear_filter(image, filter, scale))
                    image_elapsed = time.time() - image_start_time
                    runtime_list.append(image_elapsed)
                self._image_sets[return_name] = new_image_list
                batch_elapsed = time.time() - batch_start_time
                avg_runtime = statistics.mean(runtime_list)
                stats = {
                    "entire_batch_runtime": batch_elapsed,
                    "avg_image_runtime": avg_runtime,
                }
                stats_data = [function_name, stats]
                self._statistics.append(stats_data)

            elif function_name == "median_filter":
                batch_name = step["arg_batch_name"]
                return_name = step["return_batch_name"]
                weights = step["weights"]
                image_list = self._image_sets[batch_name]
                new_image_list = []
                runtime_list = []
                for image in image_list:
                    image_start_time = time.time()
                    new_image_list.append(self._m.median_filter(image, weights))
                    image_elapsed = time.time() - image_start_time
                    runtime_list.append(image_elapsed)
                self._image_sets[return_name] = new_image_list
                batch_elapsed = time.time() - batch_start_time
                avg_runtime = statistics.mean(runtime_list)
                stats = {
                    "entire_batch_runtime": batch_elapsed,
                    "avg_image_runtime": avg_runtime,
                }
                stats_data = [function_name, stats]
                self._statistics.append(stats_data)

            elif function_name == "edge_detect":
                batch_name = step["arg_batch_name"]
                return_mag_name = step["return_magnitude_batch_name"]
                return_dir_name = step["return_direction_batch_name"]
                method = step["method"]
                image_list = self._image_sets[batch_name]
                new_mag_image_list = []
                new_dir_image_list = []
                runtime_list = []
                for image in image_list:
                    image_start_time = time.time()
                    mag, dir = self._m.edge_detect(image, method)
                    new_mag_image_list.append(mag)
                    new_dir_image_list.append(dir)
                    image_elapsed = time.time() - image_start_time
                    runtime_list.append(image_elapsed)
                self._image_sets[return_mag_name] = new_mag_image_list
                self._image_sets[return_dir_name] = new_dir_image_list
                batch_elapsed = time.time() - batch_start_time
                avg_runtime = statistics.mean(runtime_list)
                stats = {
                    "entire_batch_runtime": batch_elapsed,
                    "avg_image_runtime": avg_runtime,
                }
                stats_data = [function_name, stats]
                self._statistics.append(stats_data)

            elif function_name == "dilation":
                batch_name = step["arg_batch_name"]
                return_name = step["return_batch_name"]
                strel = step["strel"]
                hot_x = step["hot_x"]
                hot_y = step["hot_y"]
                image_list = self._image_sets[batch_name]
                new_image_list = []
                runtime_list = []
                for image in image_list:
                    image_start_time = time.time()
                    new_image_list.append(self._m.dilation(image, strel, hot_x, hot_y))
                    image_elapsed = time.time() - image_start_time
                    runtime_list.append(image_elapsed)
                self._image_sets[return_name] = new_image_list
                batch_elapsed = time.time() - batch_start_time
                avg_runtime = statistics.mean(runtime_list)
                stats = {
                    "entire_batch_runtime": batch_elapsed,
                    "avg_image_runtime": avg_runtime,
                }
                stats_data = [function_name, stats]
                self._statistics.append(stats_data)

            elif function_name == "erosion":
                batch_name = step["arg_batch_name"]
                return_name = step["return_batch_name"]
                strel = step["strel"]
                hot_x = step["hot_x"]
                hot_y = step["hot_y"]
                image_list = self._image_sets[batch_name]
                new_image_list = []
                runtime_list = []
                for image in image_list:
                    image_start_time = time.time()
                    new_image_list.append(self._m.erosion(image, strel, hot_x, hot_y))
                    image_elapsed = time.time() - image_start_time
                    runtime_list.append(image_elapsed)
                self._image_sets[return_name] = new_image_list
                batch_elapsed = time.time() - batch_start_time
                avg_runtime = statistics.mean(runtime_list)
                stats = {
                    "entire_batch_runtime": batch_elapsed,
                    "avg_image_runtime": avg_runtime,
                }
                stats_data = [function_name, stats]
                self._statistics.append(stats_data)

            elif function_name == "histogram_thresh":
                batch_name = step["arg_batch_name"]
                return_name = step["return_batch_name"]
                image_list = self._image_sets[batch_name]
                new_image_list = []
                runtime_list = []
                for image in image_list:
                    image_start_time = time.time()
                    new_image_list.append(self._m.histogram_thresh(image))
                    image_elapsed = time.time() - image_start_time
                    runtime_list.append(image_elapsed)
                self._image_sets[return_name] = new_image_list
                batch_elapsed = time.time() - batch_start_time
                avg_runtime = statistics.mean(runtime_list)
                stats = {
                    "entire_batch_runtime": batch_elapsed,
                    "avg_image_runtime": avg_runtime,
                }
                stats_data = [function_name, stats]
                self._statistics.append(stats_data)

            elif function_name == "k_means_clustering":
                batch_name = step["arg_batch_name"]
                return_name = step["return_batch_name"]
                k = step["k"]
                use_loc = step["use_loc"]
                image_list = self._image_sets[batch_name]
                new_image_list = []
                runtime_list = []
                for image in image_list:
                    image_start_time = time.time()
                    new_image_list.append(self._m.k_means_clustering(image, k, use_loc))
                    image_elapsed = time.time() - image_start_time
                    runtime_list.append(image_elapsed)
                self._image_sets[return_name] = new_image_list
                batch_elapsed = time.time() - batch_start_time
                avg_runtime = statistics.mean(runtime_list)
                stats = {
                    "entire_batch_runtime": batch_elapsed,
                    "avg_image_runtime": avg_runtime,
                }
                stats_data = [function_name, stats]
                self._statistics.append(stats_data)

            elif function_name == "extract_features":
                batch_name = step["arg_batch_name"]
                dataset_name = step["dataset_name"]
                prefix = step["file_prefix"]
                preprefix = prefix[0:3]
                label = prefix
                if preprefix == "cyl":
                    label = "Columnar Epithelial"
                elif preprefix == "int":
                    label = "Intermediate Squamous Epithelial"
                elif preprefix == "let":
                    label = "Mild Nonkeratinizing Dysplastic"
                elif preprefix == "mod":
                    label = "Moderate Nonkeratinizing Dysplastic"
                elif preprefix == "par":
                    label = "Parabasal Squamous Epithelial"
                elif preprefix == "sup":
                    label = "Superficial Squamous Epithelial"
                elif preprefix == "sva":
                    label = "Severe Nonkeratinizing Dysplastic"
                image_list = self._image_sets[batch_name]
                dataset = self._datasets.get(
                    dataset_name,
                    pd.DataFrame(
                        columns=[
                            "Area",
                            "Perimeter",
                            "Median",
                            "Standard Deviation",
                            "Label",
                        ]
                    ),
                )
                runtime_list = []
                for image in image_list:
                    image_start_time = time.time()
                    features = self._m.extract_features(image)
                    features.append(label)
                    dataset.loc[len(dataset)] = features
                    image_elapsed = time.time() - image_start_time
                    runtime_list.append(image_elapsed)
                self._datasets[dataset_name] = dataset
                batch_elapsed = time.time() - batch_start_time
                avg_runtime = statistics.mean(runtime_list)
                stats = {
                    "entire_batch_runtime": batch_elapsed,
                    "avg_image_runtime": avg_runtime,
                }
                stats_data = [function_name, stats]
                self._statistics.append(stats_data)

            elif function_name == "load_dataset":
                read_path = self._config["output_datasets_path"]
                file_name = step["file_name"]
                dataset_name = step["dataset_name"]
                in_path = os.path.join(read_path, file_name + ".csv")
                in_path = Path(in_path)
                dataset = pd.read_csv(in_path)
                self._datasets[dataset_name] = dataset

            elif function_name == "save_dataset":
                save_path = self._config["output_datasets_path"]
                file_name = step["file_name"]
                dataset_name = step["dataset_name"]
                dataset = self._datasets[dataset_name]
                out_path = os.path.join(save_path, file_name + ".csv")
                out_path = Path(out_path)
                dataset.to_csv(out_path, index=False)

            elif function_name == "k_nearest_neighbors":
                dataset_name = step["dataset_name"]
                k = step["k"]
                n_fold = 10
                dataset = self._datasets[dataset_name]
                shuffled = dataset.sample(frac=1)
                groups = np.array_split(shuffled, n_fold)
                accuracy_list =[]
                for i in range(len(groups)):
                    validation = groups[i]
                    train_list = groups.copy()
                    del train_list[i]
                    training = pd.concat(train_list)
                    accuracy = self._k_nearest_neighbors(k, training, validation)
                    accuracy_list.append(accuracy)
                avg_accuracy = statistics.fmean(accuracy_list)
                print("with k = " + str(k) + ", accuracy = " + str(avg_accuracy))

            else:
                raise ValueError(
                    "'function' string was invalid!\
                         Please choose one of the options in batch.py."
                )

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
        gray_img = gray_img.astype(np.uint8)
        return gray_img

    def _mean_square_quantization_error(self, gray_img, quant_img):
        error = quant_img - gray_img
        square_error = np.square(error)
        mean_square_error = np.mean(square_error)
        return mean_square_error

    def _k_nearest_neighbors(self, k, train, valid):
        train = train.reset_index(drop=True)
        valid = valid.reset_index(drop=True)
        correct = 0
        total = 0
        for i, v_sample in valid.iterrows():
            valid_data = v_sample[:-1]
            train_data = train.iloc[:, :-1]
            diff_data = np.array(train_data) - np.array(valid_data)
            diff_data = diff_data.astype(float)
            distance_data = np.linalg.norm(diff_data, axis=1)
            near_i = distance_data.argsort()[:k]
            nearest = list(train.iloc[near_i]["Label"])
            count = Counter(nearest)
            predicted = count.most_common(1)[0][0]
            actual = v_sample[-1]
            if predicted == actual:
                correct += 1
            total += 1
        accuracy = correct/total
        return accuracy

    # def _euclidean_distance(self, list1, list2):
    #     d = 0
    #     for pair in zip(list1, list2):
    #         diff = pair[0] - pair[1]
    #         square = diff ** 2
    #         d += square
    #     root = math.sqrt(d)


if __name__ == "__main__":
    main()
