import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.layers.normalization import BatchNormalization
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils.vis_utils import plot_model
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications import VGG16

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


# Constants and Hyperparameters
size_VGG16 = (224, 224, 3)
BATCH_SIZE = 32
weight_decay = 0.001

# Data Directories
train_dir = '../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train'
test_dir = '../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/test'

# Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=5,
    width_shift_range=0.2,
    height_shift_range=0.1,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

input_size = (224, 224)
images_train = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=input_size,
    class_mode='categorical',
    subset='training',
    batch_size=BATCH_SIZE
)

images_validation = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=input_size,
    class_mode='categorical',
    subset='validation',
    batch_size=BATCH_SIZE
)

images_test = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=input_size,
    class_mode='categorical',
    batch_size=BATCH_SIZE
)

# Model Architecture
base_model = VGG16(input_shape=size_VGG16, include_top=False, weights="imagenet")

model = Sequential()
model.add(base_model)
model.add(Flatten())

# Fully connected layers
dense_units = [128, 64, 32, 32]
for units in dense_units:
    model.add(Dense(units, activation='relu', kernel_regularizer= tf.python.keras.regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

model.add(Dense(4, activation='softmax'))

# Model Summary
model.summary()

# Data Visualization
def visualize_images(folder, num_images=5):
    plt.figure(figsize=(20, 20))
    for i in range(num_images):
        file = random.choice(os.listdir(folder))
        image_path = os.path.join(folder, file)
        img = mpimg.imread(image_path)
        ax = plt.subplot(1, num_images, i + 1)
        ax.title.set_text(file)
        plt.imshow(img)

# Visualizing images from different classes
folders = [os.path.join(test_dir, class_folder) for class_folder in os.listdir(test_dir)]
for folder in folders:
    visualize_images(folder)

# Plotting Model Architecture
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
