"""
Customized MRI dataset

This work was completed in Udacity's AI for Healthcare Nanodegree Program
"""

import torch
from torch.utils.data import Dataset

class SlicesDataset(Dataset):
    """
    Customized MRI dataset
    """
    def __init__(self, data):
        """
        Args:
            data - a list of dicts {"image": image, "seg": label, "filename": f}
        """

        self.data = data

        self.slices = []

        for i, d in enumerate(data):
            for j in range(d["image"].shape[0]):
                # i: index of the volume
                # j: index of the slice
                self.slices.append((i, j))

    def __getitem__(self, idx):
        """
        Get the idx-th item from the dataset
        Args:
            idx - index of the dataset
        Output:
            sample - a dict with keys ("id", "image", "seg")
        """
        slc = self.slices[idx]
        # slc = (i, j) = (index in self.data, index in self.data[i]["image"])
        sample = dict()
        sample["id"] = idx

        sample["image"] = torch.tensor(self.data[slc[0]]["image"][slc[1]][None, :])
        sample["seg"] = torch.tensor(self.data[slc[0]]["seg"][slc[1]][None, :])

        return sample

    def __len__(self):
        """
        Return the size of the dataset
        """
        return len(self.slices)
