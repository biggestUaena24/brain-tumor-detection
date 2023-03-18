[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10312886&assignment_repo_type=AssignmentRepo)

# Abstract

## Automated Brain Tumor Detection in MRI Images Using Machine Learning Techniques

> Datasets used in this project are from: https://datasetsearch.research.google.com/
>
> This project uses the following dataset:
> https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

## Introduction

Brain tumor detection from MRI images is a critical and challenging task in medical diagnosis,
as early and accurate identification of tumors can significantly improve patient outcomes. Manual
analysis of MRI images is time-consuming and subject to inter-observer variability, highlighting the need
for automated approaches to assist medical practitioners and enhance the efficiency of their workflow.

## Objective

This project aims to develop a machine learning-based system capable of detecting brain tumors in MRI images
with high accuracy, which could potentially enable the general public to upload their MRI scans and receive
feedback on the presence of tumors.

## Methodology

To be decided

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
       ├── features       <- Scripts to turn raw data into features for modeling
       │   └── build_features.py
       │
       ├── models         <- Scripts to train models and then use trained models to make
       │   │                 predictions
       │   ├── predict_model.py
       │   └── train_model.py
       │
       └── visualization  <- Scripts to create exploratory and results oriented visualizations
           └── visualize.py
