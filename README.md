# Hippocampal Volume Quantification in Alzheimer's Progression

*Lasted updated: 12/23/2021*

[![hippocampus-frontpage-photo-crop.png](https://i.postimg.cc/5yLDXD2V/hippocampus-frontpage-photo-crop.png)](https://postimg.cc/y36L5QRQ)
<p align="center">
    Segmentation of the hippocampus structures using U-Net (images adapted from the Medical Decathlon Competition dataset)
</p>

## Project Summary

1. Build a hippocampal segmentation AI algorithm (HippoVolume.AI) by U-Net to quantify the volumes of (anterior and posterior) hippocampus
2. Model runs the segmentation task on brain MRI NIFTI files
3. Model achieves Dice score = 0.896 and Jaccard index = 0.812 on the validation dataset
4. Model is integrated into a simulated clinical network
5. Files from PACS are analyzed automatically by the AI algorithm and a report of segmentation result is generated for qualified healthcare professionals to read

[![OHIF-report-screenshot.png](https://i.postimg.cc/XqVzxqwC/OHIF-report-screenshot.png)](https://postimg.cc/DJM6w2pv)
<p align="center">
    Hippocampal segmentation tool is integrated into a simulated clinical network
</p>

## Dataset

The MRI images in this project is from the [Medical Decathlon Competition Hippocampus Dataset](http://medicaldecathlon.com/). MRI images are cropped first around the location of the hippocampus to reduce the size of the image data and the corresponding computational cost. Each volume contains the raw pixel value data of the image and the label of the segmentation as the ground truth (0: background, 1: anterior hippocampus, 2: posterior hippocampus).

We split the images into

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

We use 2 metrics to evaluate the performance of the model:

1.



[![scores-boxplots.png](https://i.postimg.cc/kG13fPDJ/scores-boxplots.png)](https://postimg.cc/jn7FS9zm)


[![tensorboard-screenshot-images.png](https://i.postimg.cc/rsGQg1mq/tensorboard-screenshot-images.png)](https://postimg.cc/ZvR85ypQ)





## Integrating into a Clinical Network

## Future Plans

To further validate the robustness of HippoVolume.AI's performance on clinical data, we plan to use the MRI stored in our PACS to test the model. All regulations will be met as advised by the legal team. Patients' participation consents will
be collected and their medical and privacy data will be protected and handled properly in accordance with the laws.
The actual true labels of the anterior and posterior hippocampus will be obtained by manual labeling by the radiology experts and clinicians together. After running HippoVolume.AI on the MRI images, quantitative metrics (Dice score and
Jaccard) will be calculated and compared to the performance on the Decathlon dataset. A copy of the report will be sent to another group of experts to determine the algorithm performance qualitatively.
To ensure the fairness of the model, analyses will be performed by the Aequitas package to assess if there are any demographic biases.



## References

1. Medical Decathlon Competition (http://medicaldecathlon.com/)
2. U-Net paper (https://arxiv.org/abs/1505.04597)
3.
