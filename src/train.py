import torch
from torch import nn

from model import CNN
from dataloader import create_dataloaders


EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, data_loader, loss_f, optimizer, device):

    # set the current mode of model into "train"
    model.train()

    # initialize current epoch's running loss, number of correct predictions, and number of total samples for training
    train_running_loss = 0.0
    train_correct = 0
    train_total = 0

    # for each images and labels...
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


def validate(model, data_loader, loss_f, device):

    # set the current mode of model into "eval"
    model.eval()

    # initialize current epoch's running loss, number of correct predictions, and number of total samples for validation
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    # disable the gradient computation
    with torch.no_grad():
        # similar logic as the for loop from train_one_epoch but without the backpropagation and optimizer executions
        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)

            scores = model(images)
            loss = loss_f(scores, labels)

            # accumulate current batch's running loss
            val_running_loss += loss.item() * images.size(0)

            # select the class with the highest score
            preds = scores.argmax(dim=1)

            # count correct predictions
            val_correct += (preds == labels).sum().item()

            # count processed samples
            val_total += labels.size(0)

    # compute the average loss and accuracy of the validating process for the current epoch
    val_loss = val_running_loss / val_total
    val_accuracy = val_correct / val_total

    return val_loss, val_accuracy



# initialize the model and move into the current device
model = CNN().to(DEVICE)

# initialize train, validation, and test dataloaders (set batch size into 8)
train_loader, val_loader, test_loader = create_dataloaders(batch_size=BATCH_SIZE)

# initialize the loss function and the optimizer (use 0.001 as the optimizer's learning rate)
loss_f = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# start the training loop
for epoch in range(EPOCHS):

    train_loss, train_accuracy = train_one_epoch(model, train_loader, loss_f, optimizer, DEVICE)
    val_loss, val_accuracy = validate(model, val_loader, loss_f, DEVICE)

    # print out current epoch
    print(f"Epoch: {epoch + 1} / {EPOCHS}")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print("-" * 30)