import numpy as np
import yaml

if __name__ == "__main__":
    config_dict = {}
    config_dict["input_image_path"] = "./Cancerous cell smears"
    config_dict["output_image_path"] = "./Output images"
    config_dict["output_statistics_path"] = "./Output statistics"
    processing_steps = []
    # Processing Step Format:
    # The following dictionary key/value pairs can be added to a step dict, which is appended to the list processing_steps. You only need to add the keys that are relevant to a given function
    # "function": "function_name" # This is the function name of the function to be called
    # "arg_batch_name": "batch1" this is the key to use to store that functions input batch name
    # "return_batch_name": "batch1" # This is the dictionary key to store that function's batched return values in
    # "file_prefix": "cyl" # This is the string that exactly matches the first characters of the file names to be loaded or saved
    # "channel": "R" # This is the channel to use in converting the RGB input image to greyscale. Can be "R", "G", "B", or None (which averages all channels)
    # "ratio": 0.5 # This is the ratio of normal pixels to salt-and-pepper noise pixels. Must be a float in interval [0, 1]
    # "mean": 0 # float, mean of gaussian noise to add to image
    # "std": 10 # float, must be positive, standard deviation of noise to add to image
    # "thresholds": [0, 10, 100, 200, 256] # List of threshold edges for use in quantizing. Must be increasing, and first and last value must be 0 and 256 respectively
    # "filter": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) # 2D numpy array of weights for linear filter. Must all be integers
    # "scale": 9 # scaling factor for linear filter (divide each fliter sum by this value)
    # "weights": np.array([[1, 2, 1], [2, 3, 2], [1, 2, 1]]) # 2D numpy array of weights for median filter. Must all be nonnegative integers
    step = {}
    step["function"] = "gaussian_noise"
    step["arg_batch_name"] = "batch1"
    step["return_batch_name"] = "batch2"
    step["mean"] = 0
    step["std"] = 10
    processing_steps.append(step)
    config_dict["processing_steps"] = processing_steps

    config_file_to_save = "./example_config.yaml"
    with open(config_file_to_save, "w") as file:
        yaml.dump(config_dict, file)
