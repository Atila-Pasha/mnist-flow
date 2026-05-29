import torch

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)


def accuracy(correct, total):

    return 100 * correct / total


def save_model(model, path):

    torch.save(
        model.state_dict(),
        path
    )


def plot_loss_curves(
    train_losses,
    test_losses
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        train_losses,
        label="Train Loss"
    )

    plt.plot(
        test_losses,
        label="Test Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Loss Curve")

    plt.legend()

    plt.grid(True)

    plt.show()


def plot_accuracy_curves(
    train_accuracies,
    test_accuracies
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        train_accuracies,
        label="Train Accuracy"
    )

    plt.plot(
        test_accuracies,
        label="Test Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy (%)")

    plt.title("Accuracy Curve")

    plt.legend()

    plt.grid(True)

    plt.show()


def plot_confusion_matrix(
    true_labels,
    predictions
):

    cm = confusion_matrix(
        true_labels,
        predictions
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=range(10)
    )

    disp.plot(ax=ax)

    plt.title(
        "MNIST Confusion Matrix"
    )

    plt.show()