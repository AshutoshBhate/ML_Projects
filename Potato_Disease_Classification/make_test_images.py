import os
import shutil

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf

IMAGE_SIZE = 256
SEED = 12
OUTPUT_DIR = "Test_Images"

dataset = keras.utils.image_dataset_from_directory(
    "PlantVillage",
    shuffle=False,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=None,
)
file_paths = dataset.file_paths

total_images = len(file_paths)
train_size = int(0.8 * total_images)
validation_size = int(0.1 * total_images)

positions = tf.data.Dataset.range(total_images)
positions = positions.shuffle(total_images, seed=SEED, reshuffle_each_iteration=False)
test_positions = positions.skip(train_size + validation_size)

copied = 0
for position in test_positions:
    source_path = file_paths[int(position)]
    class_name = os.path.basename(os.path.dirname(source_path))

    class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)
    shutil.copy2(source_path, class_dir)
    copied += 1

print("Test images copied:", copied)
for class_name in sorted(os.listdir(OUTPUT_DIR)):
    print(class_name, len(os.listdir(os.path.join(OUTPUT_DIR, class_name))))
