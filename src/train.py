import yaml

import torch
import torch.nn as nn
import torch.optim as optim

from model import MLP

from dataset import get_dataloaders

from utils import (
    accuracy,
    save_model,
    plot_loss_curves,
    plot_accuracy_curves,
    plot_confusion_matrix
)


with open(
    "configs/config.yaml",
    "r"
) as file:

    config = yaml.safe_load(file)


BATCH_SIZE = config["batch_size"]

LEARNING_RATE = config["learning_rate"]

EPOCHS = config["epochs"]

INPUT_SIZE = config["input_size"]

HIDDEN_1 = config["hidden_1"]

HIDDEN_2 = config["hidden_2"]

NUM_CLASSES = config["num_classes"]


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")


train_loader, test_loader = (
    get_dataloaders(BATCH_SIZE)
)


model = MLP(
    INPUT_SIZE,
    HIDDEN_1,
    HIDDEN_2,
    NUM_CLASSES
).to(device)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


train_losses = []

test_losses = []

train_accuracies = []

test_accuracies = []


all_preds = []

all_labels = []


for epoch in range(EPOCHS):

    # =====================
    # TRAIN
    # =====================

    model.train()

    train_loss = 0

    train_correct = 0

    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            dim=1
        )

        train_total += labels.size(0)

        train_correct += (
            predicted == labels
        ).sum().item()

    epoch_train_loss = (
        train_loss /
        len(train_loader)
    )

    epoch_train_acc = accuracy(
        train_correct,
        train_total
    )

    train_losses.append(
        epoch_train_loss
    )

    train_accuracies.append(
        epoch_train_acc
    )

    # =====================
    # TEST
    # =====================

    model.eval()

    test_loss = 0

    test_correct = 0

    test_total = 0

    all_preds = []

    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            test_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                dim=1
            )

            test_total += labels.size(0)

            test_correct += (
                predicted == labels
            ).sum().item()

            all_preds.extend(
                predicted.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    epoch_test_loss = (
        test_loss /
        len(test_loader)
    )

    epoch_test_acc = accuracy(
        test_correct,
        test_total
    )

    test_losses.append(
        epoch_test_loss
    )

    test_accuracies.append(
        epoch_test_acc
    )

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Loss: {epoch_train_loss:.4f} | "
        f"Train Acc: {epoch_train_acc:.2f}% | "
        f"Test Loss: {epoch_test_loss:.4f} | "
        f"Test Acc: {epoch_test_acc:.2f}%"
    )


save_model(
    model,
    "checkpoints/mnist_mlp.pth"
)

print(
    "\nModel saved successfully."
)


plot_loss_curves(
    train_losses,
    test_losses
)

plot_accuracy_curves(
    train_accuracies,
    test_accuracies
)

plot_confusion_matrix(
    all_labels,
    all_preds
)