# A-Deep-Learning-Approach-to-Skin-Cancer-Diagnosis

This project aims to detect Fake News. We considered three types of model.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Deployment](#deployment)
- [GitHub Workflow Development Process](#gitHub-Workflow-development-process)
- [Future Work](#future-work)
- [License](#license)

## Introduction

Fake news is one of the most common desiformation in the world and early detection is crucial for successful treatment. This project focuses on the detection of fake news, using a deep learning-based approach.

The model used in this project is a convolutional neural network (CNN), which is trained on a dataset of skin images. The CNN can take an image of a skin lesion as input and output the probability of the lesion being one of the three types of skin cancer.

## Installation
To install the required packages, please use the following command: pip install -r requirements.txt
Install:
0. python.exe -m pip install --upgrade pip
1. pip install tensorflow
2. pip install keras
3. pip install fastapi
4. pip install uvicorn
5. pip install --upgrade setuptools wheel pip
6. pip install numpy Pillow
7. pip install python-multipart
8. pip install flask
9. pip install torch torchvision torchaudio
10.pip install transformers


## Dataset
The dataset used in this project is a dataset, gave for the teacher.
The dataset was preprocessed and split into training, validation, and test sets. The images were resized and normalized to improve the performance of the model.


## Model Architecture
The model architecture used in this project is a CNN with the following layers:

- Input layer
- Data preprocessing layer (resizing and rescaling)
- Convolutional layers
- Max-pooling layers
- Flatten layer
- Fully connected layers

## Training
To train the model, follow these steps:

Download the dataset using the instructions mentioned in the Dataset section.

Run the following command: python train.py --data_path <path-to-dataset>
-
The trained model will be saved in the models directory.

## Evaluation
To evaluate the model, follow these steps:

Download the dataset using the instructions mentioned in the Dataset section.

Run the following command: python evaluate.py --data_path <path-to-dataset>

The evaluation results will be printed on the console.

## Results
The model achieved an accuracy of 71% and 85% on the test set, which is not optimal due to the small size of the dataset.

## Deployment
The Fake News detection system was deployed using a FastAPI web application, which allows users to upload images and text of news. The model was packaged as a Python module and loaded into the FastAPI server. The server returns the prediction. The code was committed to GitHub. The model was deployed successfully, and the API is accessible from a public URL.

## GitHub Workflow Development Process

## Future Work
There are several areas where this project can be extended in the future:

- Better score
- Train the model on larger datasets to improve performance.
- Develop a user-friendly interface for the system.

## License
This project is licensed under the Apache License - see the LICENSE file for details.

