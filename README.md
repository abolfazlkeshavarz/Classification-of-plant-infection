# Plant Disease Classification Using YOLO11x

This repository contains an implementation of a binary classification model using YOLO11x to detect fungal infections in plants. The model classifies plant images into two categories: **Infected** (fungal infection present) and **Uninfected** (healthy plants). The project was developed as part of a research effort to explore deep learning applications in agricultural disease detection.

## Project Overview

The goal of this project is to leverage the YOLO11x model from Ultralytics for classifying plant images based on the presence of fungal infections. The model was trained on a custom dataset and achieves exceptional performance, with a Top-1 accuracy of 100% on the test set. This repository provides the Jupyter Notebook used for training and evaluating the model, along with instructions for reproducing the results.

### Key Features

- **Model**: YOLO11x (Ultralytics) fine-tuned for binary classification.
- **Classes**: Infected (fungal infection) and Uninfected (healthy).
- **Evaluation Metric**: Top-1 Accuracy.
- **Implementation**: Python, PyTorch, and Jupyter Notebook.
- **Hardware**: Trained on a Tesla T4 GPU with 15GB memory.

## Dataset

The dataset used for training consists of plant images labeled as either *Infected* or *Uninfected*. The dataset is structured as follows:

- **Format**: Images in `.jpg` format with corresponding labels.
- **Size**: 2400 training images and 600 test images, balanced across the two classes.
- **Resolution**: Images resized to 224x224 pixels for training.

*Note*: Due to privacy and size constraints, the dataset is not included in this repository. Users must provide their own dataset with a similar structure to train the model.


## Usage

1. Open the `Infected.ipynb` notebook in Jupyter.
2. Follow the instructions in the notebook to:
   - Prepare your dataset (ensure it follows the structure described in the notebook).
   - Train the YOLO11x model for 50 epochs.
   - Evaluate the model using Top-1 accuracy.
   - Visualize predictions on sample images.


## Results

The model achieves the following performance on the test set:

- **Top-1 Accuracy**: 100% (best model, validated across multiple epochs).
- **Training Time**: Approximately 1.37 hours on a Tesla T4 GPU.
- **Model Size**: 57.0 MB (optimized weights).

Sample predictions are visualized in the notebook, showcasing the model's ability to accurately distinguish between infected and uninfected plants.



---

*Email*: abolfazlkeshavarz1112@gmail.com
*Email*: erfantarehkar@gmail.com
