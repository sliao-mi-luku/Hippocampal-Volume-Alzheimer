"""
Evaluation metrics

This work was completed in Udacity's AI for Healthcare Nanodegree Program
"""

import numpy as np

def Dice3d(a, b):
    """
    Calculates the Dice similarity score
    Arg:
        a - 3D numpy array (predicted segmentation)
        b - 3D numpy array (true segmentation)
    Output:
        dice_score: the Dice similarity coefficient
    """
    # check the dimensions
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")
    # check the shapes
    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # intersection of a and b
    intersect = np.sum(np.logical_and(a, b))
    # size of a
    A = np.sum(np.count_nonzero(a))
    # size of b
    B = np.sum(np.count_nonzero(b))
    # calculate Dice score
    dice_score = 2*intersect/(A + B)

    return dice_score


def Jaccard3d(a, b):
    """
    Calculates the Jaccard index
    Arg:
        a - 3D numpy array (predicted segmentation)
        b - 3D numpy array (true segmentation)
    Output:
        jaccard_score: the Jaccard index
    """
    # check the dimensions
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")
    # check the shapes
    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # intersection of a and b
    intersect = np.sum(np.logical_and(a, b))
    # union of a and b
    union = np.sum(np.logical_or(a, b))
    # Jaccard index
    jaccard_score = intersect/union

    return jaccard_score
