import os
import argparse
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import math

# Default dimensions of images
img_width, img_height = 256, 256

top_model_weights_path = ""
train_data_dir = "data/train"
validation_data_dir = "data/validation"
data_type = ""

# Number of epochs to train top model
epochs = 50
# Batch size used by flow_from_directory and predict_generator
batch_size = 16


def save_bottleneck_features():
    # Build the VGG16 network
    model = VGG16(include_top=False, weights="imagenet")

    datagen = ImageDataGenerator(rescale=1.0 / 255)

    def extract_features(directory, sample_count):
        generator = datagen.flow_from_directory(
            directory,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False,
        )

        predict_size = int(math.ceil(sample_count / batch_size))
        features = model.predict(generator, predict_size)
        return features

    def save_features(features, filename):
        np.save(filename, features)

    train_features = extract_features(train_data_dir, len(os.listdir(train_data_dir)))
    save_features(
        train_features, f"oasis_cross-sectional_features_train_{data_type}.npy"
    )

    validation_features = extract_features(
        validation_data_dir, len(os.listdir(validation_data_dir))
    )
    save_features(
        validation_features,
        f"oasis_cross-sectional_features_validation_{data_type}.npy",
    )


def train_top_model():
    datagen_top = ImageDataGenerator(rescale=1.0 / 255)

    def data_generator(directory):
        return datagen_top.flow_from_directory(
            directory,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,
        )

    train_generator = data_generator(train_data_dir)
    validation_generator = data_generator(validation_data_dir)

    num_classes = len(train_generator.class_indices)
    np.save(
        f"oasis_cross-sectional_class_indices_{data_type}.npy",
        train_generator.class_indices,
    )

    train_data = np.load(f"oasis_cross-sectional_features_train_{data_type}.npy")
    train_labels = to_categorical(train_generator.classes, num_classes=num_classes)

    validation_data = np.load(
        f"oasis_cross-sectional_features_validation_{data_type}.npy"
    )
    validation_labels = to_categorical(
        validation_generator.classes, num_classes=num_classes
    )

    model = Sequential(
        [
            Flatten(input_shape=train_data.shape[1:]),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        train_data,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(validation_data, validation_labels),
    )

    model.save_weights(top_model_weights_path)

    eval_loss, eval_accuracy = model.evaluate(
        validation_data, validation_labels, batch_size=batch_size, verbose=1
    )

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))

    plt.figure(1)

    plt.subplot(211)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")

    plt.subplot(212)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")

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
