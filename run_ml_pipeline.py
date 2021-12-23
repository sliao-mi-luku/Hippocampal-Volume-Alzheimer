"""
This file contains code that will start training and testing processes

This file is the work derived from Udacity's AI for Healthcare Nanodegree Program
"""

import os
import json

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData

import random

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = f"./hippocampus_data"
        self.n_epochs = 3
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = f"./test_results"

if __name__ == "__main__":
    # Get configuration
    c = Config()

    # Load data
    print("Loading data...")

    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)
    # data = LoadHippocampusData("hippocampus_data", y_shape=64, z_shape=64)
    # data[i] = {"image": image, "seg": label, "filename": f}

    # data split
    keys = range(len(data))
    split = dict()
    n = len(data)
    keys = list(keys)

    random.shuffle(keys)

    # split into 60%, 20%, 20%
    split["train"] = keys[:int(n*0.6)]
    split["val"] = keys[int(n*0.6):int(n*0.8)]
    split["test"] = keys[int(n*0.8):]

    # set up and run experiment
    exp = UNetExperiment(c, split, data)

    del data

    # run training
    exp.run()

    # prep and run testing
    results_json = exp.run_test()

    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))
