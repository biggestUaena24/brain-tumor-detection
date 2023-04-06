[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10312886&assignment_repo_type=AssignmentRepo)

# Abstract

## Automated Brain Tumor Detection in MRI Images Using Machine Learning Techniques

> Datasets used in this project are from: https://datasetsearch.research.google.com/
>
> This project uses the following dataset:
>
> https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
>
> https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
>
> The report you can find it in [here](reports/final_project_report/\_book/Deep-Learning-for-Predicting-Brain-Tumor-From-MRI-Images.pdf)
>
> The link for the presentations is: https://www.canva.com/design/DAFfLNy9-9E/Da7IHMFRXJKDKKowhEvC9A/edit?utm_content=DAFfLNy9-9E&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
>
> The link for the video presentation is: https://drive.google.com/file/d/1lGtc4JmllZH0EmvwGzqr5y-aURuVlM9t/view?usp=share_link

## Introduction

Brain tumor detection from MRI images is a critical and challenging task in medical diagnosis,
as early and accurate identification of tumors can significantly improve patient outcomes. Manual
analysis of MRI images is time-consuming and subject to inter-observer variability, highlighting the need
for automated approaches to assist medical practitioners and enhance the efficiency of their workflow.

## Objective

This project aims to develop a machine learning-based system capable of detecting brain tumors in MRI images
with high accuracy, which could potentially enable the general public to upload their MRI scans and receive
feedback on the presence of tumors. It is important to develop such a program because as addressed in introduction,
it is time-consuming for manual detection so by using such a model, it can increase and maximize efficiency of
brain tumor detection for both hospitality infrastructure and patients.

## Methodology

The first algorithm that we are going to implement is VGG-19. The benefit of using this algorithm is that it
can train data without data augmentation and further to tune the model using data augmentation. VGG-19 is consists of
19 layers which includes 16 convolutional layers, 3 fully connected layers, and 5 max-pooling layers, and some benefits
of using VGG-19 is that it is a pre-trained model containing million of images and thousand of classes. This means
that the model has already learned a wide range of features and it can be fined-tune for some specific tasks
and in this case, classifying brain tumor. It is also one of the models that has a strong feature extraction and
has a fairly simple architecture. However, VGG-19 is also computational heavy and also takes up large memory which
means it is harder for deployment with limited memory devices.

The second algorithm that we are implementing is ResNet50. The main benefit of using ResNet50 is the use of
residual connections, which enable the network to learn identity functions. This helps alleviate the vanishing
gradient problem and allows the network to train efficiently even with a large number of layers. It can also be scaled up
or down depending on the specific problem and available computational resources. It also takes up lesser memory than
VGG architecture.

# Environment Setup

## Pre-requisites

- Miniconda/Anaconda 3

## Create Conda Environment

Run the following command on a terminal prompt.

```
conda env create -f environment.yml
```

# Project Instructions

## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README for describing highlights for using this ML project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    |       |__ testing
    |       |__ training
    |
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   └── README.md      <- Youtube Video Link
    │   └── final_project_report <- final report .pdf format and supporting files
    │   └── presentation   <-  final power point presentation
    |
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
       ├── __init__.py    <- Makes src a Python module
       │
       ├── preprocessing data           <- Scripts to download or generate data and pre-process the data
       │   └── make_dataset.py
       │   └── pre-processing.py
       │
       │
       ├── models         <- Scripts to train models and then use trained models to make
           │                 predictions
           ├── resnet_model.py
           └── vgg19_model.py
