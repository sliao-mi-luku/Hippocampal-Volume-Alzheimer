"""
Load the hippocampus dataset

This work was completed in Udacity's AI for Healthcare Nanodegree Program

Source of data: https://github.com/udacity/nd320-c3-3d-imaging-starter/tree/master/data/TrainingSet

"""

import os
from os import listdir
from os.path import isfile, join

import numpy as np
from medpy.io import load

from utils.utils import med_reshape

def LoadHippocampusData(root_dir, y_shape, z_shape):
    '''
    Load the dataset and reshape the images
    Args:
        root_dir - path to the images and labels folders
        y_shape - final size of the image height. default=64
        z_shape - final size of the image width. default=64
    Output:
        out: a list of dicts. Each dict have keys "image", "seg", and "filename"
    '''

    # path to the images
    image_dir = os.path.join(root_dir, 'images')
    # path to the labels
    label_dir = os.path.join(root_dir, 'labels')

    # a list of paths to all images
    images = [f for f in listdir(image_dir) if (isfile(join(image_dir, f)) and f[0] != ".")]

    out = []

    for f in images:
        image, _ = load(os.path.join(image_dir, f))
        label, _ = load(os.path.join(label_dir, f))

        # normalize
        image_max_val = np.amax(image)
        image = np.true_divide(image, image_max_val)

        # reshape the images and labels
        image = med_reshape(image, new_shape=(image.shape[0], y_shape, z_shape))
        label = med_reshape(label, new_shape=(label.shape[0], y_shape, z_shape)).astype(int)

        # append to output
        out.append({"image": image, "seg": label, "filename": f})

    print(f"Processed {len(out)} files, total {sum([x['image'].shape[0] for x in out])} slices")

    return np.array(out)
