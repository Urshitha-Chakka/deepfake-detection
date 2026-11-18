# Deepfake Detection Using Deep Learning

This project develops a deepfake image classification system using a custom Convolutional Neural Network (CNN) and, in a later stage, transfer learning.

The goal is to classify facial images as either **real** or **deepfake** and compare the performance of a CNN trained from scratch with a pretrained transfer-learning model.

## Project Objective

The main research question is:

**How accurately can deep learning models distinguish authentic facial images from deepfake images, and can transfer learning improve performance compared with a custom CNN?**

## Dataset

The project uses a preprocessed image dataset derived from **Celeb-DF V2**.

The original Celeb-DF V2 dataset is video-based. In the preprocessed version used here:

- frames were extracted from real and fake videos
- facial regions were detected using dlib's frontal face detector
- cropped faces were enhanced using ESPCN-based super-resolution
- images were resized to 256 x 256 pixels

The local dataset contains approximately:

- 50,360 real images
- 50,671 fake images
- 101,031 images in total

The full image dataset is not included in this GitHub repository because of its size.

## Data Leakage Audit

The original train, validation, and test folders contained substantial source-video overlap.

The initial audit found:

- 5,741 source video IDs appearing in more than one split
- 682 fake identity pairs appearing in more than one split

This could cause data leakage because frames from the same source video may appear during both training and testing.

To address this, the dataset was repartitioned using grouped splitting.

For real images, all frames from the same source video were kept within a single split.

For fake images, all frames associated with the same fake identity pair were kept within a single split.

After rebuilding the partitions:

- source video overlap across clean splits: 0
- fake identity-pair overlap across clean splits: 0

The final split is approximately:

| Split | Fake | Real |
|---|---:|---:|
| Train | 40,619 | 40,375 |
| Validation | 4,873 | 4,965 |
| Test | 5,179 | 5,020 |

## Exploratory Data Analysis

The dataset was checked for:

- class balance
- train, validation, and test distribution
- image dimensions
- missing image paths
- corrupted image files
- image color mode
- visual differences between real and fake samples
- pixel-intensity distributions
- unique source-video counts

Key findings:

- the dataset is approximately balanced between real and fake images
- sampled images have dimensions of 256 x 256 pixels
- no corrupted images were detected
- no missing image paths were found
- sampled images were stored in RGB format

## Current Model

A baseline custom CNN has been created with:

- input size: 256 x 256 x 3
- training-only data augmentation
- four convolutional blocks
- max pooling
- global average pooling
- dense layer with ReLU activation
- dropout
- sigmoid output for binary classification

The model is compiled using:

- Adam optimizer
- binary cross-entropy loss
- accuracy
- precision
- recall
- AUC

Callbacks include:

- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint

## Project Structure

```text
deepfake-detection/
|
|-- data/
|   |-- celeb_df_manifest.csv
|   |-- celeb_df_clean_manifest.csv
|
|-- notebook/
|   |-- 01_dataset_audit.ipynb
|   |-- 02_eda.ipynb
|   |-- 03_custom_cnn.ipynb
|
|-- models/
|
|-- result/
|
|-- .gitignore
|-- README.md
```

## Notebook Description

### `01_dataset_audit.ipynb`

Contains:

- dataset structure inspection
- filename parsing
- manifest creation
- source-video analysis
- leakage detection
- grouped train, validation, and test split creation
- clean manifest generation

### `02_eda.ipynb`

Contains:

- class-distribution analysis
- image-integrity checks
- image dimension checks
- RGB verification
- sample image visualization
- pixel-intensity analysis
- source-video distribution analysis

### `03_custom_cnn.ipynb`

Contains:

- TensorFlow setup
- manifest loading
- label encoding
- TensorFlow data pipeline
- batching and prefetching
- data augmentation
- custom CNN architecture
- model compilation
- callbacks
- initial training setup

## Development Environment

The project has been developed locally using:

- Python
- Anaconda
- Jupyter Notebook
- TensorFlow / Keras
- Pandas
- NumPy
- Matplotlib
- Pillow
- Scikit-learn

The local machine contains an Intel Arc 140V GPU, but the current TensorFlow environment does not detect it as an available TensorFlow GPU device.

Local testing and development are therefore performed on CPU.

## Current Status

Completed:

- dataset acquisition
- dataset audit
- manifest creation
- leakage detection
- leakage-safe splitting
- exploratory data analysis
- TensorFlow setup
- input pipeline
- data augmentation
- baseline custom CNN architecture
- model compilation

In progress:

- CNN training and evaluation

Planned:

- full custom CNN training
- confusion matrix and classification metrics
- ROC-AUC evaluation
- error analysis
- transfer-learning model
- comparison between custom CNN and transfer learning

## Dataset Availability

The full image dataset is not included in this repository because of its size.

Users should download the Celeb-DF V2-derived preprocessed image dataset separately and place it in:

```text
data/Celeb_V2/
```

The repository includes manifest files that document the image labels and leakage-safe data partitions.

## Future Work

The next stages of the project will include:

1. completing custom CNN training
2. evaluating test-set performance
3. analyzing misclassified images
4. implementing transfer learning
5. comparing both approaches using:
   - accuracy
   - precision
   - recall
   - F1-score
   - ROC-AUC
   - confusion matrices
