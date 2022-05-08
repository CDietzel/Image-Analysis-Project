import yaml

if __name__ == "__main__":
    config_dict = {}
    config_dict["input_image_path"] = "./Cancerous cell smears"
    config_dict["output_image_path"] = "./Output images"
    config_dict["output_statistics_path"] = "./Output statistics"
    config_dict["output_datasets_path"] = "./Output datasets"
    config_dict["histogram_path"] = "./Output histograms"
    processing_steps = []
    # Processing Step Format:
    # The following dictionary key/value pairs can be added to a step dict, which is appended to the list processing_steps. You only need to add the keys/values that are relevant to a given function
    # "function": "function_name" # This is the function name of the function to be called
    # "arg_batch_name": "batch1" this is the key to use to store that functions input batch name
    # "return_batch_name": "batch1" # This is the dictionary key to store that function's batched return values in
    # "file_prefix": "cyl" # This is the string that exactly matches the first characters of the file names to be loaded or saved
    # "channel": "R" # This is the channel to use in converting the RGB input image to greyscale. Can be "R", "G", "B", or None (which averages all channels)
    # "ratio": 0.5 # This is the ratio of normal pixels to salt-and-pepper noise pixels. Must be a float in interval [0, 1]
    # "mean": 0 # float, mean of gaussian noise to add to image
    # "std": 10 # float, must be positive, standard deviation of noise to add to image
    # "thresholds": [0, 10, 100, 200, 256] # List of threshold edges for use in quantizing. Must be increasing, and first and last value must be 0 and 256 respectively
    # "filter": [[1, 1, 1], [1, 1, 1], [1, 1, 1]] # 2D array of weights for linear filter. Must all be integers
    # "scale": 9 # scaling factor for linear filter (divide each fliter sum by this value)
    # "weights": [[1, 2, 1], [2, 3, 2], [1, 2, 1]] # 2D array of weights for median filter. Must all be nonnegative integers
    # "method": "prewitt", "sobel", or "compass" # edge detection operator to use.
    # "strel": [[0, 1, 0], [1, 1, 1], [0, 1, 0]] # 2d array of "0" or "1". Structuring element to use for erosion or dilation
    # "hot_x": 1 # x-index of the hot spot for the structuring element in erosion or dilation
    # "hot_y": 1 # y-index of the hot spot for the structuring element in erosion or dilation
    # "bin_thresh": 128 # pixel intensity to use for binary thresholding.
    # "k": 2 # Number of cluster centers to use for K-means clustering
    # "use_loc": True # Choose whether to include pixel x-y location in k-means clustering calculations

    prefix_list = ["cyl", "inter", "let", "mod", "para", "super", "svar"]
    # This second prefix list will only process the first image from each batch, to save computation time
    # prefix_list = ["cyl01", "inter01", "let01", "mod01", "para01", "super01", "svar01"]
    # prefix_list = ["cyl01"]

    for prefix in prefix_list:

        step = {}
        step["function"] = "load_image_set"
        step["return_batch_name"] = "batch1"
        step["file_prefix"] = prefix
        step["channel"] = None
        processing_steps.append(step)

        step = {}
        step["function"] = "edge_detect"
        step["arg_batch_name"] = "batch1"
        step["return_magnitude_batch_name"] = "batch2"
        step["return_direction_batch_name"] = "batch3"
        step["method"] = "prewitt"
        processing_steps.append(step)

        step = {}
        step["function"] = "edge_detect"
        step["arg_batch_name"] = "batch1"
        step["return_magnitude_batch_name"] = "batch4"
        step["return_direction_batch_name"] = "batch5"
        step["method"] = "sobel"
        processing_steps.append(step)

        step = {}
        step["function"] = "edge_detect"
        step["arg_batch_name"] = "batch1"
        step["return_magnitude_batch_name"] = "batch6"
        step["return_direction_batch_name"] = "batch7"
        step["method"] = "compass"
        processing_steps.append(step)

        step = {}
        step["function"] = "histogram_thresh"
        step["arg_batch_name"] = "batch1"
        step["return_batch_name"] = "batch8"
        processing_steps.append(step)

        step = {}
        step["function"] = "dilation"
        step["arg_batch_name"] = "batch8"
        step["return_batch_name"] = "batch9"
        step["strel"] = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        step["hot_x"] = 1
        step["hot_y"] = 1
        processing_steps.append(step)

        step = {}
        step["function"] = "erosion"
        step["arg_batch_name"] = "batch8"
        step["return_batch_name"] = "batch10"
        step["strel"] = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        step["hot_x"] = 1
        step["hot_y"] = 1
        processing_steps.append(step)

        step = {}
        step["function"] = "k_means_clustering"
        step["arg_batch_name"] = "batch1"
        step["return_batch_name"] = "batch11"
        step["k"] = 5
        step["use_loc"] = False
        processing_steps.append(step)

        step = {}
        step["function"] = "k_means_clustering"
        step["arg_batch_name"] = "batch1"
        step["return_batch_name"] = "batch12"
        step["k"] = 5
        step["use_loc"] = True
        processing_steps.append(step)

        step = {}
        step["function"] = "save_image_set"
        step["arg_batch_name"] = "batch1"
        step["file_prefix"] = prefix + "_grayscale"
        processing_steps.append(step)

        step = {}
        step["function"] = "save_image_set"
        step["arg_batch_name"] = "batch2"
        step["file_prefix"] = prefix + "_prewitt_edge_mag"
        processing_steps.append(step)

        step = {}
        step["function"] = "save_image_set"
        step["arg_batch_name"] = "batch3"
        step["file_prefix"] = prefix + "_prewitt_edge_dir"
        processing_steps.append(step)

        step = {}
        step["function"] = "save_image_set"
        step["arg_batch_name"] = "batch4"
        step["file_prefix"] = prefix + "_sobel_edge_mag"
        processing_steps.append(step)

        step = {}
        step["function"] = "save_image_set"
        step["arg_batch_name"] = "batch5"
        step["file_prefix"] = prefix + "_sobel_edge_dir"
        processing_steps.append(step)

        step = {}
        step["function"] = "save_image_set"
        step["arg_batch_name"] = "batch6"
        step["file_prefix"] = prefix + "_compass_edge_mag"
        processing_steps.append(step)

        step = {}
        step["function"] = "save_image_set"
        step["arg_batch_name"] = "batch7"
        step["file_prefix"] = prefix + "_compass_edge_dir"
        processing_steps.append(step)

        step = {}
        step["function"] = "save_image_set"
        step["arg_batch_name"] = "batch8"
        step["file_prefix"] = prefix + "_histogram_thresh"
        processing_steps.append(step)

        step = {}
        step["function"] = "save_image_set"
        step["arg_batch_name"] = "batch9"
        step["file_prefix"] = prefix + "_dilation"
        processing_steps.append(step)

        step = {}
        step["function"] = "save_image_set"
        step["arg_batch_name"] = "batch10"
        step["file_prefix"] = prefix + "_erosion"
        processing_steps.append(step)

        step = {}
        step["function"] = "save_image_set"
        step["arg_batch_name"] = "batch11"
        step["file_prefix"] = prefix + "_k_means_no_loc"
        processing_steps.append(step)

        step = {}
        step["function"] = "save_image_set"
        step["arg_batch_name"] = "batch12"
        step["file_prefix"] = prefix + "_k_means_with_loc"
        processing_steps.append(step)

    config_dict["processing_steps"] = processing_steps

    config_file_to_save = "./config.yaml"
    with open(config_file_to_save, "w") as file:
        yaml.dump(config_dict, file, default_flow_style=None)
