import torch
import numpy as np
from torch.utils.data import DataLoader
import os
import nibabel as nib
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Define the top model
num_classes = 2  # Update this based on your dataset
model = nn.Sequential(
    nn.AdaptiveAvgPool3d((1, 1, 1)),  # Pool across spatial dimensions
    nn.Flatten(),
    nn.Linear(1, 16),  # Adjust input size based on flattened size
    nn.Linear(16, 64),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, num_classes),
    nn.Sigmoid(),
)

# Define the transformations used during training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # Assuming grayscale image
])

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

        # Extract the middle slice (assuming the depth dimension is the first dimension)
        middle_slice_index = img_data.shape[0] // 2
        middle_slice = img_data[middle_slice_index]

        if self.transform:
            middle_slice = self.transform(middle_slice)

        return middle_slice



def prepare_sample_data(data_dir):
    # Prepare sample data loader
    dataset = NiftiDataset(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    return loader

# def test_model(model, data_loader):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()

#     predictions = []
#     with torch.no_grad():
#         for inputs, _ in data_loader:
#             inputs = inputs.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             predictions.extend(predicted.cpu().numpy())
#     return predictions

def test_model(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    return predictions

def load_labels(dataset_dir):
    class_labels = {}
    for class_index, class_name in enumerate(os.listdir(dataset_dir)):
        class_labels[class_name] = class_index
    return class_labels

def load_data_with_labels(dataset_dir):
    class_labels = load_labels(dataset_dir)
    data = []
    labels = []
    for class_name, class_index in class_labels.items():
        class_dir = os.path.join(dataset_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith('.nii'):
                data.append(os.path.join(class_dir, filename))
                labels.append(class_index)
    return data, labels

def analyze_predictions(predictions, true_labels):
    # Convert the lists to numpy arrays for easier computation
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Compute accuracy
    correct_predictions = (predictions == true_labels).sum()
    total_predictions = len(predictions)
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    # Compute confusion matrix
    num_classes = len(np.unique(true_labels))
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(true_labels, predictions):
        confusion_matrix[true_label, pred_label] += 1

    print("Confusion Matrix:")
    print(confusion_matrix)



if __name__ == "__main__":
    # Set paths and parameters
    model_path = "trained_resnet50_model.pth"  # Update with your model path
    sample_data_dir = "sample_data/train/RAW"  # Update with your sample data directory

    # Load pre-trained model
    model.load_state_dict(torch.load(model_path))

    # Prepare sample data loader
    sample_data_loader = prepare_sample_data(sample_data_dir)

    # Test the model on sample data
    predictions = test_model(model, sample_data_loader)
    print("Prediction:")
    print(predictions)

    data, labels = load_data_with_labels(sample_data_dir)
    # print("Data files:")
    # print(data)
    print("Labels:")
    print(labels)

    # Analyze predictions (e.g., accuracy, confusion matrix, etc.)
    analyze_predictions(predictions,labels)