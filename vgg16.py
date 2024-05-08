# -*- coding: utf-8 -*-
"""VGG16.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Ocq0o-vHbdnGrWRkJ2gv7cI1V-54ZJkz
"""

import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


import os
import shutil

def delete_folder(folder_path):
    """
    Delete a folder and its contents if the folder exists.

    Parameters:
        folder_path (str): Path to the folder to be deleted.
    """
    if os.path.exists(folder_path):
        # Use shutil.rmtree to delete the folder and its contents recursively
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and its contents have been deleted.")
    else:
        print(f"Folder '{folder_path}' does not exist.")

def generate_test_labels(test_dir):
    """
    Generate labels for the test data based on the subdirectory names in the test directory.

    Parameters:
        test_dir (str): Path to the directory containing the test data.

    Returns:
        numpy.ndarray: Array of labels for the test data.
    """
    # Get the class labels from the subdirectory names
    class_labels = sorted(os.listdir(test_dir))

    # Create a dictionary to map class labels to numerical indices
    label_to_index = {label: i for i, label in enumerate(class_labels)}

    # Initialize an empty list to store the labels
    test_labels = []

    # Iterate through the subdirectories in the test directory
    for subdir in class_labels:
        subdir_path = os.path.join(test_dir, subdir)
        if os.path.isdir(subdir_path):
            # Get the label index for the current subdirectory
            label_index = label_to_index[subdir]
            # Get the number of files (instances) in the subdirectory
            num_instances = len(os.listdir(subdir_path))
            # Append the label index to the test_labels list for each instance in the subdirectory
            test_labels.extend([label_index] * num_instances)

    # Convert the list of labels to a numpy array
    test_labels = np.array(test_labels)

    return test_labels

def organize_images_by_class(sample_dir, selected_image_paths):
    """
    Organize selected images into subdirectories based on their classes.

    Parameters:
        sample_dir (str): Path to the directory where images will be organized.
        selected_image_paths (list): List of paths to the selected images.

    Returns:
        None
    """
    # Organize images into subdirectories based on classes
    for image_path in selected_image_paths:
        class_name = os.path.basename(os.path.dirname(image_path))
        class_dir = os.path.join(sample_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        # Modify to copy .jpg files instead of .nii files
        if filename.endswith('.jpg'):
            dst = os.path.join(class_dir, filename)
            shutil.copy(image_path, dst)

def load_image_paths(train_dir):
    """
    Load image paths from the specified directory.

    Parameters:
        train_dir (str): Path to the directory containing image classes.

    Returns:
        list: A list of paths to all the images in the directory.
    """
    all_image_paths = []
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        print(class_dir)
        image_paths = [os.path.join(class_dir, filename) for filename in os.listdir(class_dir)]
        all_image_paths.extend(image_paths)
    return all_image_paths


# Directory containing data
train_dir = 'data/train'
val_dir = 'data/validation'
test_dir = 'data/test'

print(train_dir)
print(val_dir)
print(test_dir)

# Load filenames
all_train_images = load_image_paths(train_dir)
all_validate_images = load_image_paths(val_dir)
all_test_images = load_image_paths(test_dir)

# # Select a random sample of 300 images
# selected_train_image_paths = np.random.choice(all_train_images, size=300, replace=False)
# selected_validate_image_paths = np.random.choice(all_validate_images, size=300, replace=False)
# selected_test_image_paths = np.random.choice(all_test_images, size=300, replace=False)

# # Create a directory to store the sampled images
# sample_train_dir = '/content/sampled_train'
# sample_validate_dir = '/content/sampled_validate'
# sample_test_dir = '/content/sampled_test'
# delete_folder(sample_train_dir)
# delete_folder(sample_validate_dir)
# delete_folder(sample_test_dir)
# os.makedirs(sample_train_dir, exist_ok=True)
# os.makedirs(sample_validate_dir, exist_ok=True)
# os.makedirs(sample_test_dir, exist_ok=True)

# organize_images_by_class(sample_train_dir, selected_train_image_paths)
# organize_images_by_class(sample_validate_dir, selected_validate_image_paths)
# organize_images_by_class(sample_test_dir, selected_validate_image_paths)

# print(selected_train_image_paths[:10])  # Display the first 10 paths

# import os

# sample_train_dir = '/content/sampled_train'

# # List all files and directories in the sampled_train directory
# contents = os.listdir(sample_train_dir)

# # Print the list of contents
# print(contents)

#import shutil

# Define the path to the folder you want to delete
#folder_path = '/content/train/train/train'

# Use shutil.rmtree() to recursively remove the folder and its contents
#shutil.rmtree(folder_path)

# Constants and Hyperparameters
size_VGG16 = (224, 224, 3)
BATCH_SIZE = 32
weight_decay = 0.001
LEARNING_RATE = 0.001
NUM_CLASSES = 3

# Use the sampled images for training
train_dir = train_dir
val_dir = val_dir
test_dir = test_dir

# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=size_VGG16[:2],
    class_mode='categorical',
    batch_size=BATCH_SIZE
)

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=size_VGG16[:2],
    class_mode='categorical',
    batch_size=BATCH_SIZE
)

validation_generator = val_datagen.flow_from_directory(
    directory=val_dir,
    target_size=size_VGG16[:2],
    class_mode='categorical',
    batch_size=BATCH_SIZE
)

# Model Architecture
base_model = VGG16(input_shape=size_VGG16, include_top=False, weights="imagenet")

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    #steps_per_epoch=30,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# Plot training history
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.show()