# Golf-Swing-Phase-Classifier

## Project Overview

The goal of this project is to build a convolutional neural network from scratch using PyTorch to classify four phases of a golf swing: address, top, impact, and finish.

The project focuses on understanding the complete computer vision workflow, including creating a custom image dataset, splitting the data at the swing level, building and training a CNN, validating model performance, and evaluating trained models on a held-out test set.

Rather than relying on a pretrained architecture, the model was designed manually to better understand the fundamental structure and training process of convolutional neural networks.

## Dataset

The dataset used in this project is available on Kaggle: [Golf Swing Phase Image Dataset](https://www.kaggle.com/datasets/mkim2178/golf-swing-phase-image-dataset)

Download the dataset and place the `data` directory in the project root.

Here's the expected directory structure:
```text
data/
├── address/
├── top/
├── impact/
└── finish/
```

## How to Run

### Install Dependencies

This project was developed with Python 3.11.

It is recommended to create and activate a virtual environment before installing the dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Train the Model
Run the training script:
```bash
python src/train.py
```

The best model checkpoint for the selected random seed will be saved in the `models` directory (the default random seed is `42`).

To train a model with a different initialization, modify `RANDOM_SEED` in `train.py`.

### Test a Trained Model
In `test.py`, set `MODEL_SEED` to the seed number of an existing trained model in the `models` directory, then run:
```bash
python src/test.py
```

The script will report the test loss and accuracy and visualize correct and incorrect predictions.