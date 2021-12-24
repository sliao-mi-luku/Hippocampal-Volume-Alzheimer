# Hippocampal Volume Quantification in Alzheimer's Progression

*Lasted updated: 12/23/2021*

[![hippocampus-frontpage-photo-crop.png](https://i.postimg.cc/5yLDXD2V/hippocampus-frontpage-photo-crop.png)](https://postimg.cc/y36L5QRQ)
<p align="center">
    Segmentation of the hippocampus structures using U-Net (images adapted from the Medical Decathlon Competition dataset)
</p>

## Project Summary

1. Build a hippocampal segmentation AI algorithm, HippoVolume.AI, by U-Net to quantify the volumes of (anterior and posterior) hippocampus
2. Model runs the segmentation task on brain MRI NIFTI files
3. Model achieves Dice score = 0.896 and Jaccard index = 0.812 on the validation dataset
4. Model is integrated into a simulated clinical network
5. Files from PACS are analyzed automatically by the AI algorithm and a report of segmentation result is generated for qualified healthcare professionals to read

[![OHIF-report-screenshot.png](https://i.postimg.cc/XqVzxqwC/OHIF-report-screenshot.png)](https://postimg.cc/DJM6w2pv)
<p align="center">
    Hippocampal segmentation tool is integrated into a simulated clinical network
</p>

## Introduction

**HippoVolume.AI** is a radiological AI system that estimates the hippocampal volume from brain MRI data. The propose of this AI system is to serve as a tool to aid qualified healthcare professionals to assess the progression of the Alzheimer's disease (AD). By quantitatively estimating the hippocampal volume of patients by HippoVolume.AI at their every visit, decreasing in the hippocampal volume can be detected to help with early detection of AD. After further assessments by the clinicians, treatments or therapies can be done to alleviate the progression.

HippoVolume.AI can be integrated into the clinical network to read the data from PACS directly and automatically runs the algorithm on the relevant volumes that contains the hippocampus structure. Locations of the posterior and anterior hippocampus are automatically segmented and a report will be generated and saved to the PACS for clinicians to access. The report shows the volumes of the anterior, posterior, and total hippocampus volumes and put the predicted segmentation labels on the original images for easier interpretation from the standard OHIF viewer. An example of the report can be seen above.

## Dataset

The MRI images in this project is from the [Medical Decathlon Competition Hippocampus Dataset](http://medicaldecathlon.com/). MRI images are cropped first around the location of the hippocampus to reduce the size of the image data and the corresponding computational cost. Each volume contains the raw pixel value data of the image and the label of the segmentation as the ground truth (0: background, 1: anterior hippocampus, 2: posterior hippocampus).

We split the images into a training datasets consisting of 156 series (5,500 MRI slices) and a validation set of 52 series (1,870 MRI slices).

[![example-data.png](https://i.postimg.cc/NfkBbC1M/example-data.png)](https://postimg.cc/w7MCjV48)
<p align="center">
    Examples of images and the corresponding labels from Medical Decathlon Competition Hippocampus Dataset
</p>

## Model

We use the [U-Net](https://arxiv.org/abs/1505.04597) ([GitHub](https://github.com/MIC-DKFZ/basic_unet_example)) for the segmentation. Models with similar structure have been successfully applied on the biomedical images segmentation such as cell tracking.

The model is trained on 156 series (5,500 slices of MRIs) and validated on a separate set of 52 series (1,870 slices of MRIs) for 3 epochs. We stop the training after 3 epochs because the loss is converging (see the figures below).

[![tensorboard-screenshot-losses-crop.png](https://i.postimg.cc/tJr91xj4/tensorboard-screenshot-losses-crop.png)](https://postimg.cc/dZTbpD3b)
<p align="center">
    Loss during training and validation. The loss converges after 3 epochs
</p>

## Evaluation

We use 2 metrics to evaluate the performance of the model: **Dice coefficient** and **Jaccard index**.

- **Dice coefficient** (Dice score) = `2*INTERSECTION(volume_pred, volume_true) / (volume_pred + volume_true)`

- **Jaccard index** (Jaccard score) = `INTERSECTION(volume_pred, volume_true) / UNION(volume_pred, volume_true)`

The two scores range can from 0 (poor performance) to 1 (best performance). By running the algorithm on the validation dataset, we get the mean `Dice score = 0.8955` and mean `Jaccard scores = 0.8116` across all volumes:

[![scores-boxplots.png](https://i.postimg.cc/kG13fPDJ/scores-boxplots.png)](https://postimg.cc/jn7FS9zm)
<p align="center">
    Model performance on the validation dataset. Mean Dice and Jaccard scores: 0.8955 and 0.8116
</p>

[![tensorboard-screenshot-images.png](https://i.postimg.cc/rsGQg1mq/tensorboard-screenshot-images.png)](https://postimg.cc/ZvR85ypQ)
<p align="center">
    An example data predicted by HippoVolume.AI (bottom row) compared to the actual label (middle row)
</p>


## Integrating into a Clinical Network

HippoVolume.AI can access the volumes in PACS (Orthanc) from the MRI scanner and estimates the hippocampal size automatically. A report with segmentation results will be saved to the PACS and is accessible by the clinicians with the viewer system such as OHIF for medical diagnoses.

[![OHIF-report-screenshot.png](https://i.postimg.cc/XqVzxqwC/OHIF-report-screenshot.png)](https://postimg.cc/DJM6w2pv)
<p align="center">
    Hippocampal segmentation tool is integrated into a simulated clinical network
</p>

## References

1. Medical Decathlon Competition (http://medicaldecathlon.com/)
2. U-Net paper (https://arxiv.org/abs/1505.04597)
3. U-Net implementation in PyTorch (https://github.com/MIC-DKFZ/basic_unet_example)
4. Udacity AI for Healthcare Nanodegree Program (https://github.com/udacity/nd320-c3-3d-imaging-starter)
