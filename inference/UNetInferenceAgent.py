"""
Hippocampal Volume Quantification

Udacity's AI in Healthcare Nanodegree

===

Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        # original size
        d0, d1, d2 = volume.shape
        # padding the size
        padded_volume = np.zeros((d0, self.patch_size, self.patch_size))
        # fill in values
        padded_volume[:d0, :d1, :d2] = volume
        # pass to the model
        pred_volume = self.single_volume_inference(padded_volume)

        return pred_volume

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        slices = []

        for i in range(volume.shape[0]):
            slice_t = torch.tensor(volume[i][None, None, :]).float().to(self.device)  # (1, 1, patch_size, patch_size)

            with torch.no_grad():
                prediction = self.model(slice_t).detach().cpu()  # (1, 3, patch_size, patch_size)
                prediction_mask = torch.squeeze(torch.argmax(prediction, dim=1))  # (patch_size, patch_size)
                slices.append(prediction_mask.tolist())

        return np.array(slices)
