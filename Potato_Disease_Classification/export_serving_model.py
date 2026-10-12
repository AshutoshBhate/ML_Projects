import os
import shutil

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers
import tensorflow as tf

IMAGE_SIZE = 256
N_CLASSES = 3
MODEL_DIR = "Saved_Models/1"

resize_rescale = keras.Sequential(
    [
        layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.Rescaling(1.0 / 255),
    ],
    name="resize_rescale",
)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ],
    name="data_augmentation",
)

model = keras.Sequential(
    [
        keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        resize_rescale,
        data_augmentation,
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(N_CLASSES, activation="softmax"),
    ],
    name="potato_disease_cnn",
)

trained_export = tf.saved_model.load(MODEL_DIR)
trained_values = {}
for variable in trained_export.variables:
    trained_values[variable.name] = variable.numpy()

for weight in model.weights:
    weight.assign(trained_values[weight.path + ":0"])

print("Weights copied:", len(model.weights))

del trained_export
shutil.rmtree(MODEL_DIR)

archive = keras.export.ExportArchive()
archive.track(model)
archive.add_endpoint(
    name="serve",
    fn=lambda images: model(images, training=False),
    input_signature=[tf.TensorSpec(shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32)],
)
archive.write_out(MODEL_DIR)
