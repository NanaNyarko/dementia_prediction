from pickletools import optimize
import torch
import torch.nn as nn
import torchvision.models as models
import os
import argparse
import numpy as np
from torchvision import transforms
import nibabel as nib
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score

# Default dimensions of images
img_depth, img_height, img_width = 256, 256, 256

top_model_weights_path = ""
train_data_dir = "data/train"
validation_data_dir = "data/validation"
train_features_file = "oasis_longitudinal_demographics_features_train.npy"
validation_features_file = "oasis_longitudinal_demographics_features_train.npy"
data_type = ""

# Number of epochs to train top model
epochs = 50
# Batch size used by DataLoader
batch_size = 16

class NiftiDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_paths = self._get_file_paths()

    def _get_file_paths(self):
        file_paths = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.nii'):
                    file_paths.append(os.path.join(root, file))
        return file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        print("Loading file:", file_path)
        img = nib.load(file_path)
        img_data = np.array(img.dataobj)

        # Normalize and scale the image data to [0, 1]
        img_data = np.clip(img_data, np.min(img_data), np.max(img_data))
        img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

        # Extract label from file path or any other method to get the label
        label = self._extract_label(file_path)
        print("Label:", label)
        label_tensor = torch.tensor([1.0, 0.0] if label == 'no_dementia' else [0.0, 1.0], dtype=torch.float32)
        print("Image Data Shape:", img_data.shape)
        print("Label Tensor:", label_tensor)

        if self.transform:
            img_data = self.transform(img_data)

        return img_data, label_tensor
    
    def _extract_label(self, file_path):
        # Extract label from the directory name just before the file name
        directory_name = os.path.basename(os.path.dirname(file_path))
        label = directory_name
        return label
         
def save_bottleneck_features():
    # Load pre-trained ResNet50 model
    model = models.resnet50(weights=None)
    
    # Download the pre-trained weights if not available locally
    if not os.path.exists('resnet50.pth'):
        print("Downloading ResNet50 model weights...")
        model = models.resnet50(weights=None)
        torch.save(model.state_dict(), 'resnet50.pth')

    # Load the downloaded model weights
    state_dict = torch.load('resnet50.pth')
    model.load_state_dict(state_dict)
    print("model resnet available")
    # Remove the fully connected layer at the end
    modules = list(model.children())[:-1]
    model = nn.Sequential(*modules)
    model.eval()  # Set the model to evaluation mode

    def extract_features(directory):
        features = []
        
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith(".nii"):
                    file_path = os.path.join(root, filename)
                    img = nib.load(file_path)
                    img_data = np.array(img.dataobj)
                    
                    # Normalize and scale the image data to uint8 range [0, 255]
                    img_data = np.clip(img_data, np.min(img_data), np.max(img_data))
                    img_data = np.uint8((img_data / np.max(img_data)) * 255)
                    
                    # Select a single 2D slice (e.g., middle slice) from the 3D NIfTI image data
                    slice_index = img_data.shape[0] // 2
                    img_slice = img_data[slice_index, :, 0]
                    
                    # Convert grayscale image to RGB
                    img_pil = Image.fromarray(img_slice, mode='L')
                    img_pil = img_pil.convert('RGB')
                    
                    # Resize the image to 244x244
                    resize_transform = transforms.Resize((244, 244))
                    img_pil = resize_transform(img_pil)

                    # Apply preprocessing transformations
                    img_tensor = transforms.ToTensor()(img_pil)
                    img_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)
                    
                    # Add batch dimension
                    img_tensor = img_tensor.unsqueeze(0)
                    
                    with torch.no_grad():
                        # Extract features from ResNet model
                        features_batch = model(img_tensor)
                        
                        # Flatten the features to (batch_size, 256)
                        features_batch = features_batch.view(features_batch.size(0), -1)
                        print("Features Batch Shape:", features_batch.shape)
                        features.append(features_batch)
        return torch.cat(features)



    train_features = extract_features(train_data_dir)
    np.save(train_features_file, train_features.numpy())

    validation_features = extract_features(validation_data_dir)
    np.save(validation_features_file, validation_features.numpy())

def evaluate_model(model, data_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_predictions = []
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            running_loss += loss.item()

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_predictions)

    return accuracy, running_loss / len(data_loader), precision, f1, auc

def train_top_model(train_loader, validation_loader, epochs=10):

    train_features = np.load(train_features_file)
    validation_features = np.load(validation_features_file)

    # Define the top model
    num_classes = 2  # Update this based on your dataset
     # Determine the input size for the first linear layer based on the actual output size of the features
    input_features = train_features.shape[1]  # Assuming train_features is available globally
    print(f"input_feature ",input_features)

    model = nn.Sequential(
        nn.AdaptiveAvgPool3d((1, 1, 1)),  # Adaptive average pooling
        nn.Flatten(),  # Flatten the input
        nn.Linear(input_features, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
        nn.Sigmoid(),
    )


    print(model)
   

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters())  # Corrected optimizer initialization

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the device

            optimizer.zero_grad()

            outputs = model(inputs.view(inputs.size(0), -1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_accuracy, _, _, _, _ = evaluate_model(model, train_loader, criterion)
        val_accuracy, val_loss, val_precision, val_f1, val_auc = evaluate_model(model, validation_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

    return model, train_losses, val_losses, train_accuracies, val_accuracies


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
        img_depth, img_height, img_width = 176, 208, 176  # Update dimensions for your dataset

    # train_data_dir = os.path.join(train_data_dir, data_type)
    train_data_dir = train_data_dir
    validation_data_dir = validation_data_dir
    top_model_weights_path = f"oasis_longitudinal_demographics_{data_type}.h5"

     # Define transformations
    transform = None  # You can define transformations if needed

    # Create datasets and dataloaders
    print(f"train_data_dir = '{train_data_dir}'")
    train_dataset = NiftiDataset(train_data_dir, transform=transform)
    validation_dataset = NiftiDataset(validation_data_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    save_bottleneck_features()
    train_top_model(train_loader, validation_loader)

