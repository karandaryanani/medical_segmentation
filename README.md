# Group 19 - MS5: Semantic Segmentation of Brain MRI Scans using Transfer Learning

## Team Members
- Karan Daryanani
- Zahra Rasouli
- Marium Tapal

## Overview
This project focuses on semantic segmentation of brain MRI scans, utilizing the Brain Tumor Segmentation (BRATS) 2020 dataset. We employ transfer learning techniques in conjunction with a U-Net architecture, aiming to achieve high performance in loss, accuracy, and the Intersection over Union (IoU) score.

## Dataset Description
The BRATS 2020 dataset features multi-institutional routine clinically-acquired pre-operative multimodal MRI scans of glioblastoma (GBM/HGG) and lower-grade glioma (LGG). The scans are provided in four modalities: T1, T1Gd (post-contrast T1-weighted), T2, and T2-FLAIR. These scans have been annotated for various tumor sub-regions by expert neuroradiologists.

### Key Features
- Multi-institutional MRI scans with expert-level annotations.
- Labels for GD-enhancing tumor, peritumoral edema, and necrotic/non-enhancing tumor core.
- Pre-processed scans co-registered to the same anatomical template, interpolated to 1 mm^3 resolution, and skull-stripped.

### Accessing the Data
To access the BRATS 2020 data, participants must register following the instructions on the official BRATS 2020 "[Registration/Data Request](https://www.med.upenn.edu/sbia/brats2020/registration.html)" page.

## Repository Contents
- `group19-MS5.ipynb`: The main Jupyter notebook containing our data analysis, model training, and evaluation.
- Additional scripts and utility files as applicable.

## Getting Started
1. Clone this repository.
2. Ensure you have the required Python environment and dependencies. (See `requirements.txt`)
3. Download the BRATS 2020 dataset following the provided instructions and place it in the designated directory.
4. Run `group19-MS5.ipynb` for a detailed walkthrough of our analysis and modeling.

### Overview of notebook:
- Libraries
- Problem Statement
- EDA and Visualization
- Data Preparation
- Early Model Attempts
- Final Model
- Results and Inference
- Conclusion and Future Work

### Libraries
This project uses libraries such as TensorFlow, Keras, NumPy, Matplotlib, Pandas, and Scikit-Learn. Additionally, we utilize nibabel for handling NIfTI files and other image processing libraries.

### Problem Statement
The primary challenge is to accurately segment brain tumors from multimodal MRI scans. Our objective is to optimize the model across three metrics: loss, accuracy, and IoU score. We leverage transfer learning on a U-Net algorithm with pretrained ImageNet weights, iteratively refining the architecture for optimal performance.

### EDA and Visualization
- Our data consists of medical images in NIfTI format, annotated for segmentation tasks.
- We analyze pixel value distributions, label distributions, and MRI modalities to understand data characteristics.
- Visualization of data includes the distribution of pixel values, example images highlighting segmentation tasks, and modalities insights.

### Data Preparation
- We process the 4D MRI data into 2D slices, considering each modality.
- The preprocessing steps include image resizing, normalization, and one-hot encoding of labels.
- We create a TensorFlow dataset pipeline with a batch size of 64 and split data into training, testing, and validation sets.

### Early Model Attempts
- Initial models are based on the U-Net architecture, modified for our specific task.
- We experiment with various learning rates (1e-2, 1e-3, 1e-4) and observe the model's performance.
- Challenges like high loss and accuracy issues lead us to further refine our model with additional layers and techniques.

### Final Model
- We enhance our model with skip connections, batch normalization, max pooling, and dropout layers to improve performance.
- The final model is compiled with an Adam optimizer and categorical crossentropy loss function.
- We utilize callbacks for early stopping based on the mean IoU metric.

### Results and Inference
- The final model shows significant improvement in accuracy, loss, and IoU compared to earlier models.
- We provide detailed visualizations of the training history for loss, accuracy, and IoU.
- The model's ability to accurately segment brain tumors is demonstrated through predictions on test image slices.

### Conclusion and Future Work
- We discuss the strengths and limitations of U-Net and other architectures like Mask R-CNN, SegNet, and FastFCN.
- Our exploration suggests potential improvements and future directions, such as experimenting with newer architectures like Gated Shape CNN for enhanced segmentation accuracy.

## Acknowledgments
Special thanks to the BRATS 2020 Organizers, and the CS 209B Class for providing us with the tools to do all of this.
