"""
Training, validation, and testing routines of the segmentation task

This work was completed in Udacity's AI for Healthcare Nanodegree Program
"""

import os
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_prep.SlicesDataset import SlicesDataset
from utils.utils import log_to_tensorboard
from utils.volume_stats import Dice3d, Jaccard3d
from networks.RecursiveUNet import UNet
from inference.UNetInferenceAgent import UNetInferenceAgent

class UNetExperiment:
    """
    Segmentation task with UNet (https://arxiv.org/abs/1505.04597)
    """
    def __init__(self, config, split, dataset):
        self.n_epochs = config.n_epochs
        self.split = split
        self._time_start = ""
        self._time_end = ""
        self.epoch = 0
        self.name = config.name

        # create output folders
        dirname = f'{time.strftime("%Y-%m-%d_%H%M", time.gmtime())}_{self.name}'
        self.out_dir = os.path.join(config.test_results_dir, dirname)
        os.makedirs(self.out_dir, exist_ok=True)

        # create data loaders
        self.train_loader = DataLoader(SlicesDataset(dataset[split["train"]]),
                batch_size=config.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(SlicesDataset(dataset[split["val"]]),
                batch_size=config.batch_size, shuffle=True, num_workers=0)

        # test data
        self.test_data = dataset[split["test"]]

        if not torch.cuda.is_available():
            print("WARNING: No CUDA device is found. This may take significantly longer!")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ceate U-Net
        self.model = UNet(num_classes=3)
        self.model.to(self.device)

        # loss
        self.loss_function = torch.nn.CrossEntropyLoss()

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        # scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

        # set up tensorboard
        self.tensorboard_train_writer = SummaryWriter(comment="_train")
        self.tensorboard_val_writer = SummaryWriter(comment="_val")

    def train(self):
        """
        Train the model
        """

        print(f"Training epoch {self.epoch}...")

        # train mode
        self.model.train()

        # iterate over batches of the training dataset
        for i, batch in enumerate(self.train_loader):
            # batch is a dict, with keys: "id", "image", "seg"
            # batch["id"] = [id_0, id_1, ..., id_batch_size]
            # batch["image"].shape = (batch_size, 1, patch_size, patch_size)
            # batch["seg"].shape = (batch_size, 1, patch_size, patch_size)

            self.optimizer.zero_grad()

            data = batch["image"].float().to(self.device)  # (batch_size, 1, patch_size, patch_size)
            target = batch["seg"].long().to(self.device)  # (batch_size, 1, patch_size, patch_size)

            prediction = self.model(data)  # (batch_size, 3, patch_size, patch_size)

            prediction_softmax = F.softmax(prediction, dim=1)  # (batch_size, 3, patch_size, patch_size)

            loss = self.loss_function(prediction, target[:, 0, :, :])
            loss.backward()

            self.optimizer.step()

            if (i % 10) == 0:
                # Output to console on every 10th batch
                print(f"\nEpoch: {self.epoch} Train loss: {loss}, {100*(i+1)/len(self.train_loader):.1f}% complete")

                counter = 100*self.epoch + 100*(i/len(self.train_loader))

                log_to_tensorboard(self.tensorboard_train_writer,
                                   loss,
                                   data,
                                   target,
                                   prediction_softmax,
                                   prediction,
                                   counter)

            print(".", end='')

        print("\nTraining complete")



    def validate(self):
        """
        Validation process
        """

        print(f"Validating epoch {self.epoch}...")

        # eval mode
        self.model.eval()

        loss_list = []

        with torch.no_grad():
            # iterate over batches in the validation dataset
            for i, batch in enumerate(self.val_loader):
                data = batch["image"].float().to(self.device)
                target = batch["seg"].long().to(self.device)

                with torch.no_grad():
                    prediction = self.model(data).detach()  # (batch_size, 3, patch_size, patch_size)
                    prediction_softmax = F.softmax(prediction, dim=1)  # (batch_size, 3, patch_size, patch_size)
                    loss = self.loss_function(prediction, target[:, 0, :, :])

                print(f"Batch {i}. Data shape {data.shape} Loss {loss}")

                loss_list.append(loss.item())

        self.scheduler.step(np.mean(loss_list))

        log_to_tensorboard(self.tensorboard_val_writer,
                           np.mean(loss_list),
                           data,
                           target,
                           prediction_softmax,
                           prediction,
                           (self.epoch + 1)*100)

        print(f"Validation complete")

    def save_model_parameters(self):
        """
        Saves model parameters to a file in results directory
        """
        path = os.path.join(self.out_dir, "model.pth")
        torch.save(self.model.state_dict(), path)

    def load_model_parameters(self, path=''):
        """
        Loads model weights
        """
        if not path:
            model_path = os.path.join(self.out_dir, "model.pth")
        else:
            model_path = path

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        else:
            raise Exception(f"Could not find path {model_path}")

    def run_test(self):
        """
        Test on the test dataset
        """

        print("Testing...")
        self.model.eval()

        inference_agent = UNetInferenceAgent(model=self.model, device=self.device)

        out_dict = {}
        out_dict["volume_stats"] = []
        dc_list = []
        jc_list = []

        # iterate over the batches in the test data
        for i, x in enumerate(self.test_data):
            pred_label = inference_agent.single_volume_inference(x["image"])

            # evaluation metrics (Dice and Jaccard scores)
            dc = Dice3d(pred_label, x["seg"])
            jc = Jaccard3d(pred_label, x["seg"])
            dc_list.append(dc)
            jc_list.append(jc)

            out_dict["volume_stats"].append({"filename": x['filename'], "dice": dc, "jaccard": jc})

            print(f"{x['filename']} Dice {dc:.4f}. {100*(i+1)/len(self.test_data):.2f}% complete")

        out_dict["overall"] = {"mean_dice": np.mean(dc_list), "mean_jaccard": np.mean(jc_list)}

        print("\nTesting complete.")

        return out_dict

    def run(self):
        """
        Train, validate, and test the model
        """
        self._time_start = time.time()

        print("Experiment started.")

        # iterate over epochs
        for self.epoch in range(self.n_epochs):
            # train
            self.train()
            # validate
            self.validate()

        # save model
        self.save_model_parameters()

        self._time_end = time.time()
        print(f"Run complete. Total time: {time.strftime('%H:%M:%S', time.gmtime(self._time_end - self._time_start))}")
