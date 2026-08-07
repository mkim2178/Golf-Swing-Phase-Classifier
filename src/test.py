import torch
from torch import nn
from pathlib import Path

from model import CNN
from dataloader import create_dataloaders
from train import evaluate
import matplotlib.pyplot as plt

# model seed (must match the seed number of an existing model checkpoint in the models directory)
MODEL_SEED = 42

# batch size
BATCH_SIZE = 8

# path to the saved best model
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
MODEL_PATH = PROJECT_DIR / "models" / f"best_model_seed_{MODEL_SEED}.pth"

# use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# swing-phase labels
CLASS_NAMES = ["address", "top", "impact", "finish"]

def visualize_predictions(model, data_loader, device):

    # set the model to evaluation mode
    model.eval()

    # images, labels, and predictions
    all_images = []
    all_labels = []
    all_predictions = []

    # disable gradient computation
    with torch.no_grad():

        # iterate through each batch of images and labels
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # get the class scores and select the class with the highest scores
            scores = model(images)
            predictions = scores.argmax(dim=1)

            # store every image, label, and prediction in lists for visualization and move each tensor to CPU memory
            all_images.extend(images.cpu())
            all_labels.extend(labels.cpu())
            all_predictions.extend(predictions.cpu())

    # number of rows and columns (3 * 8 = 24 test images)
    rows = 3
    cols = 8

    # init the figure and axes and flatten the axes
    fig, axes = plt.subplots(rows, cols, figsize=(16, 28))
    axes = axes.flatten()

    # iterate through the images, predictions, and labels with index from enumerating lists that contains all of images, preds, and labels
    for index, (image, prediction, label) in enumerate(zip(all_images, all_predictions, all_labels)):

        # convert the predicted and actual numeric labels into class names
        pred_cls = CLASS_NAMES[prediction.item()]
        actual_cls = CLASS_NAMES[label.item()]

        # determine whether the prediction is correct
        result = "correct" if prediction.item() == label.item() else "incorrect"

        # rearrange its dimensions from [C,H,W] to [H,W,C] for Matplotlib
        image = image.cpu().permute(1, 2, 0)

        # display the image without axes and show its prediction details
        axes[index].imshow(image)
        axes[index].axis("off")
        axes[index].set_title(
            f"Prediction: {pred_cls}\nActual: {actual_cls}\nResult: {result}",
            fontsize=7
            )
    # display the image with its prediction result
    plt.subplots_adjust(hspace=0.5)
    plt.show()
                

def main():

    # init the same model architecture used during training
    model = CNN().to(DEVICE)

    # init the test dataloader  and ignore the training and validation loaders
    _, _, test_loader = create_dataloaders(batch_size=BATCH_SIZE)

    # init the same loss function used during training
    loss_f = nn.CrossEntropyLoss()

    # load the parameters from the best saved model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # get the test loss and accuracy
    test_loss, test_accuracy = evaluate(model, test_loader, loss_f, DEVICE)

    # print final test results
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # visualize the test images and predictions
    visualize_predictions(model, test_loader, DEVICE)

if __name__ == "__main__":
    main()
