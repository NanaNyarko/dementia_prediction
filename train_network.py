import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

# Default dimensions of images
img_width, img_height = 256, 256

top_model_weights_path = ""
train_data_dir = "data/train"
validation_data_dir = "data/validation"
data_type = ""

# Number of epochs to train top model
epochs = 50
# Batch size used by DataLoader
batch_size = 16


def save_bottleneck_features():
    # Load pre-trained VGG16 model
    model = models.vgg16(pretrained=True)

    model.classifier = nn.Sequential(
        *list(model.classifier.children())[:-1]
    )  # Remove the last fully connected layer
    model.eval()  # Set the model to evaluation mode

    transform = transforms.Compose(
        [
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
        ]
    )

    def extract_features(directory):
        dataset = datasets.ImageFolder(directory, transform=transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        features = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                features_batch = model(inputs)
                features.append(features_batch)
        return torch.cat(features)

    train_features = extract_features(train_data_dir)
    np.save(
        f"oasis_cross-sectional_features_train_{data_type}.npy", train_features.numpy()
    )

    validation_features = extract_features(validation_data_dir)
    np.save(
        f"oasis_cross-sectional_features_validation_{data_type}.npy",
        validation_features.numpy(),
    )


def train_top_model():
    transform = transforms.Compose(
        [
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.ImageFolder(train_data_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    validation_dataset = datasets.ImageFolder(validation_data_dir, transform=transform)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )

    num_classes = len(train_dataset.classes)
    np.save(
        f"oasis_cross-sectional_class_indices_{data_type}.npy",
        train_dataset.class_to_idx,
    )

    print(num_classes)

    # Define the top model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512 * 7 * 7, 256),
        nn.Linear(256, num_classes),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Sigmoid(),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(validation_loader)
        val_acc = 100 * correct / total
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), history["train_loss"], label="Train Loss")
    plt.plot(range(epochs), history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), history["train_acc"], label="Train Acc")
    plt.plot(range(epochs), history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-t",
        "--type",
        required=True,
        help="type of dataset / model to train (options: FSL_SEG, PROCESSED, or RAW)",
    )
    args = vars(ap.parse_args())
    data_type = args["type"]

    if data_type == "FSL_SEG":
        img_width, img_height = 176, 208

    train_data_dir = os.path.join(train_data_dir, data_type)
    validation_data_dir = os.path.join(validation_data_dir, data_type)
    top_model_weights_path = f"oasis_cross-sectional_{data_type}.h5"

    save_bottleneck_features()
    train_top_model()
