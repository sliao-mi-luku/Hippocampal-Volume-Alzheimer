"""
Helper functions

This work was completed in Udacity's AI for Healthcare Nanodegree Program
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from PIL import Image

# Tell Matplotlib to not try and use interactive backend
mpl.use("agg")

def mpl_image_grid(images):
    """
    Create an image grid from an array of images. Show up to 16 images in one figure
    Code provided by Udacity
    Args:
        image {Torch tensor} -- NxWxH array of images
    Outputs:
        Matplotlib figure
    """
    # Create a figure to contain the plot.
    n = min(images.shape[0], 16) # no more than 16 thumbnails
    rows = 4
    cols = (n // 4) + (1 if (n % 4) != 0 else 0)
    figure = plt.figure(figsize=(2*rows, 2*cols))
    plt.subplots_adjust(0, 0, 1, 1, 0.001, 0.001)
    for i in range(n):
        # Start next subplot.
        plt.subplot(cols, rows, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if images.shape[1] == 3:
            # this is specifically for 3 softmax'd classes with 0 being bg
            # We are building a probability map from our three classes using
            # fractional probabilities contained in the mask
            vol = images[i].detach().numpy()
            img = [[[(1-vol[0,x,y])*vol[1,x,y], (1-vol[0,x,y])*vol[2,x,y], 0] \
                            for y in range(vol.shape[2])] \
                            for x in range(vol.shape[1])]
            plt.imshow(img)
        else: # plotting only 1st channel
            plt.imshow((images[i, 0]*255).int(), cmap= "gray")

    return figure


def log_to_tensorboard(writer, loss, data, target, prediction_softmax, prediction, counter):
    """
    Logs to Tensorboard. Code provided by Udacity.
    Args:
        writer {SummaryWriter} -- PyTorch Tensorboard wrapper to use for logging
        loss {float} -- loss
        data {tensor} -- image data
        target {tensor} -- ground truth label
        prediction_softmax {tensor} -- softmax'd prediction
        prediction {tensor} -- raw prediction (to be used in argmax)
        counter {int} -- batch and epoch counter
    """
    writer.add_scalar("Loss", loss, counter)
    writer.add_figure("Image Data", mpl_image_grid(data.float().cpu()), global_step=counter)
    writer.add_figure("Mask", mpl_image_grid(target.float().cpu()), global_step=counter)
    writer.add_figure("Probability map", mpl_image_grid(prediction_softmax.cpu()), global_step=counter)
    writer.add_figure("Prediction", mpl_image_grid(torch.argmax(prediction.cpu(), dim=1, keepdim=True)), global_step=counter)


def save_numpy_as_image(arr, path):
    """
    Saves the image
    Args:
        arr - 2D numpy array of the image
        path - path to store the image
    """
    plt.imshow(arr, cmap="gray")
    plt.savefig(path)


def med_reshape(image, new_shape):
    """
    Reshapes 3d array into new_shape by padding zeros to the right and bottom
    Args:
        image - 3D numpy array of the pixel values
        new_shape - a tuple (dx, dy, dz) of the final shape
    Output:
        reshaped_image - 3D numpy array of the pixel values after reshaping
    """
    # create an empty 3D array
    reshaped_image = np.zeros(new_shape)
    # shape of the original image
    d0, d1, d2 = image.shape
    # paste the values
    reshaped_image[:d0, :d1, :d2] = image

    return reshaped_image
