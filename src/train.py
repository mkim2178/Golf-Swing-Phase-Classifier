import torch
from torch import nn

from model import CNN
from dataloader import create_dataloaders


# number of epoch
epochs = 20

# current device (using CPU/GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# initialize the model and move into the current device
model = CNN().to(device)

# initialize train, validation, and test dataloaders (set batch size into 8)
train_loader, val_loader, test_loader = create_dataloaders(batch_size=8)

# initialize the loss function and the optimizer (use 0.001 as the optimizer's learning rate)
loss_f = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# start the training loop
for epoch in range(epochs):

    # set the current mode of model into "train"
    model.train()

    # print out current epoch
    print(f"Epoch: {epoch + 1} / {epochs}")

    # initialize current epoch's running loss, number of correct predictions, and number of total samples
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    # for each images and labels
    for images, labels in train_loader:
        # move images and labels into current device (CPU/GPU)
        images = images.to(device)
        labels = labels.to(device)

        # reset gradients from previous batch
        optimizer.zero_grad()

        # execute the forward pass
        scores = model(images)
        loss = loss_f(scores, labels)

        # execute the backward pass
        loss.backward()

        # update model parameters
        optimizer.step()

        # accumulate current batch's running loss
        running_loss += loss.item() * images.size(0)

        # select the class with the highest score
        preds = scores.argmax(dim=1)

        # count correct predictions 
        correct_preds += (preds == labels).sum().item()

        # count processed samples
        total_samples += labels.size(0)

    # compute the average loss and accuracy for the current epoch
    train_loss = running_loss / total_samples
    train_accuracy = correct_preds / total_samples
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")

