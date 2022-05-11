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
    # "dataset_name": "dataset" # this is the key to use to store that functions input/output dataset
    # "file_name": "dataset" # This is a string that will make up the filename of the file to save/load
    #

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
        step["function"] = "extract_features"
        step["arg_batch_name"] = "batch1"
        step["dataset_name"] = "dataset"
        step["file_prefix"] = prefix
        processing_steps.append(step)

    step = {}
    step["function"] = "save_dataset"
    step["file_name"] = "dataset"
    step["dataset_name"] = "dataset"
    processing_steps.append(step)

    step = {}
    step["function"] = "load_dataset"
    step["file_name"] = "dataset"
    step["dataset_name"] = "dataset"
    processing_steps.append(step)

    step = {}
    step["function"] = "k_nearest_neighbors"
    step["dataset_name"] = "dataset"
    step["k"] = 1
    processing_steps.append(step)

    step = {}
    step["function"] = "k_nearest_neighbors"
    step["dataset_name"] = "dataset"
    step["k"] = 3
    processing_steps.append(step)

    step = {}
    step["function"] = "k_nearest_neighbors"
    step["dataset_name"] = "dataset"
    step["k"] = 5
    processing_steps.append(step)

    step = {}
    step["function"] = "k_nearest_neighbors"
    step["dataset_name"] = "dataset"
    step["k"] = 7
    processing_steps.append(step)

    step = {}
    step["function"] = "k_nearest_neighbors"
    step["dataset_name"] = "dataset"
    step["k"] = 13
    processing_steps.append(step)

    step = {}
    step["function"] = "k_nearest_neighbors"
    step["dataset_name"] = "dataset"
    step["k"] = 21
    processing_steps.append(step)

    step = {}
    step["function"] = "k_nearest_neighbors"
    step["dataset_name"] = "dataset"
    step["k"] = 99
    processing_steps.append(step)

    config_dict["processing_steps"] = processing_steps

    config_file_to_save = "./config.yaml"
    with open(config_file_to_save, "w") as file:
        yaml.dump(config_dict, file, default_flow_style=None)
