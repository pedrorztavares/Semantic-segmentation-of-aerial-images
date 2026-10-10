# Aerial Image Semantic Segmentation

This project focuses on the semantic segmentation of aerial images using Deep Learning models.

The main goal is to automatically identify and classify different types of terrain and structures present in aerial imagery, producing a segmentation mask for each input image.

## Project Overview

A custom dataset was created specifically for this project using aerial images collected from Google Earth and manually annotated with CVAT.
This dataset is publicly available at: https://www.kaggle.com/datasets/antniobalhau/rural-aerial-images-semantic-segmentation-dataset

The segmentation task includes six classes:

- Background
- Agricultural fields
- Vegetation
- Water bodies
- Roads
- Structures

The project compares two Deep Learning architectures:

- U-Net
- DeepLabV3+

The models were trained and evaluated using semantic segmentation metrics such as Pixel Accuracy and Intersection over Union (IoU).

## Dataset

The dataset was manually created and annotated for this project.

Each aerial image has a corresponding segmentation mask where each pixel is associated with one of the defined classes.

Before training, the images are processed into patches of:

```text
256 x 256 pixels
