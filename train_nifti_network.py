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

# Default dimensions of images
img_depth, img_height, img_width = 256, 256, 256

top_model_weights_path = ""
train_data_dir = "data/train"
validation_data_dir = "data/validation"
data_type = ""

# Number of epochs to train top model
epochs = 50
# Batch size used by DataLoader
batch_size = 16

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
        print(directory)
        features = []
        
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith(".nii"):
                    file_path = os.path.join(root, filename)
                    print(file_path)
                    img = nib.load(file_path)
                    img_data = np.array(img.dataobj)
                    
                    # Normalize and scale the image data to uint8 range [0, 255]
                    img_data = np.clip(img_data, np.min(img_data), np.max(img_data))
                    img_data = np.uint8((img_data / np.max(img_data)) * 255)
                    
                    # Select a single 2D slice (e.g., middle slice) from the 3D NIfTI image data
                    slice_index = img_data.shape[0] // 2  # Select the middle slice along the depth dimension
                    img_slice = img_data[slice_index, :, 0]  # Select the slice and remove the singleton dimension
                    
                    # Convert the selected slice to an RGB PIL image
                    img_pil = Image.fromarray(img_slice, mode='L')  # Grayscale image (single channel)
                    # Convert Grayscale Image to RGB by Replicating Single Channel Across All Channels
                    img_pil = img_pil.convert('RGB')
                    
                    # Apply preprocessing transformations
                    img_pil = img_pil.resize((img_width, img_height))
                    img_tensor = transforms.ToTensor()(img_pil)
                    img_tensor = transforms.Normalize(mean=[0.485], std=[0.229])(img_tensor)  # Assuming grayscale image
                    
                    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
                    with torch.no_grad():
                        features_batch = model(img_tensor)
                        features.append(features_batch)
        return torch.cat(features)


    train_features = extract_features(train_data_dir)
    np.save(f"oasis_cross-sectional_features_train_{data_type}.npy", train_features.numpy())

    validation_features = extract_features(validation_data_dir)
    np.save(f"oasis_cross-sectional_features_validation_{data_type}.npy", validation_features.numpy())

def train_top_model():
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])

    # Define the top model
    num_classes = 2  # Update this based on your dataset
    model = nn.Sequential(
        nn.AdaptiveAvgPool3d((1, 1, 1)),  # Adjust for the output size of ResNet50
        nn.Flatten(),
        nn.Linear(2048, 256),  # Adjust input size based on ResNet50 output
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
        nn.Sigmoid(),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optimize.RMSprop(model.parameters())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        history["train_loss"].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(validation_loader)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

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

    train_data_dir = os.path.join(train_data_dir, data_type)
    validation_data_dir = os.path.join(validation_data_dir, data_type)
    top_model_weights_path = f"oasis_cross-sectional_{data_type}.h5"

    save_bottleneck_features()
    train_top_model()