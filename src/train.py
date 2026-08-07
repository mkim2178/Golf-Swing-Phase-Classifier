import torch
from torch import nn
from pathlib import Path

from model import CNN
from dataloader import create_dataloaders

# controls model weight initialization, training-data shuffling, and other PyTorch randomness
RANDOM_SEED = 42

# fixed constants
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 0.001

# train.py must be run from the src directory
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
MODEL_PATH = PROJECT_DIR / "models" / f"best_model_seed_{RANDOM_SEED}.pth"

# "cuda" will be initialized, if GPU is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, data_loader, loss_f, optimizer, device):

    # enable training model
    model.train()

    # track cumulative loss, correct predictions, and processed samples
    train_running_loss = 0.0
    train_correct = 0
    train_total = 0

    # process each batch of images and labels
    for images, labels in data_loader:

        # move images and labels into current device (CPU/GPU)
        images = images.to(device)
        labels = labels.to(device)


        # reset gradients from previous batch
        optimizer.zero_grad()

        scores = model(images)
        loss = loss_f(scores, labels)

        loss.backward()

        # update model parameters
        optimizer.step()

        # accumulate current batch's running loss
        train_running_loss += loss.item() * images.size(0)

        # select the class with the highest score
        preds = scores.argmax(dim=1)

        # count correct predictions 
        train_correct += (preds == labels).sum().item()

        # count processed samples
        train_total += labels.size(0)

    # compute the average loss and accuracy of the training process for the current epoch
    train_loss = train_running_loss / train_total
    train_accuracy = train_correct / train_total

    return train_loss, train_accuracy


def evaluate(model, data_loader, loss_f, device):

    # enable evaluation mode
    model.eval()

    # initialize running loss, number of correct predictions, and total samples
    running_loss = 0.0
    correct = 0
    total = 0

    # disable gradient computation during evaluation
    with torch.no_grad():
        # evaluate each batch without backpropagation or parameter updates
        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)

            scores = model(images)
            loss = loss_f(scores, labels)

            # accumulate current batch's running loss
            running_loss += loss.item() * images.size(0)

            # select the class with the highest score
            preds = scores.argmax(dim=1)

            # count correct predictions
            correct += (preds == labels).sum().item()

            # count processed samples
            total += labels.size(0)

    # compute the average loss and accuracy
    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def main():

    # reproducibility
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    
    
    # initialize the model and move into the current device
    model = CNN().to(DEVICE)

    # initialize train and validation dataloaders (set batch size into 8 and ignore the test dataloader)
    train_loader, val_loader, _ = create_dataloaders(batch_size=BATCH_SIZE)

    # initialize the loss function and the optimizer (use 0.001 as the optimizer's learning rate)
    loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


    # record the lowest validation loss
    best_val_loss = float("inf")

    # start the training loop
    for epoch in range(EPOCHS):

        train_loss, train_accuracy = train_one_epoch(model, train_loader, loss_f, optimizer, DEVICE)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_f, DEVICE)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Best Model saved. Validation loss: {best_val_loss:.4f}")
        
        # print out current epoch
        print(f"Epoch: {epoch + 1} / {EPOCHS}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print("-" * 30)


if __name__ == "__main__":
    main()