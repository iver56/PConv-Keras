import datetime
import gc
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from matplotlib.ticker import NullFormatter
from tqdm import tqdm

from libs.pconv_model import PConvUnet
from libs.util import random_mask

# SETTINGS
TRAIN_DIR = os.path.join("data", "training")
TEST_DIR = os.path.join("data", "test")
VAL_DIR = os.path.join("data", "validation")

BATCH_SIZE = 1

os.makedirs(os.path.join("data", "test_samples"), exist_ok=True)


class DataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory, *args, **kwargs):
        generator = super().flow_from_directory(
            directory, class_mode=None, *args, **kwargs
        )
        while True:
            # Get augmented image samples
            original_image = next(generator)

            # Get masks for each image sample
            mask = np.stack(
                [
                    random_mask(original_image.shape[1], original_image.shape[2])
                    for _ in range(original_image.shape[0])
                ],
                axis=0,
            )

            # Apply masks to all image sample
            masked = deepcopy(original_image)
            masked[mask == 0] = 1

            # Yield ([ori, mask],  ori) training batches
            # print(masked.shape, ori.shape)
            gc.collect()
            yield [masked, mask], original_image


# Create training generator
train_datagen = DataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1.0 / 255,
    horizontal_flip=False,
)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(512, 512), batch_size=BATCH_SIZE
)

# Create validation generator
val_datagen = DataGenerator(rescale=1.0 / 255)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR, target_size=(512, 512), batch_size=BATCH_SIZE, seed=1
)

# Create testing generator
test_datagen = DataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR, target_size=(512, 512), batch_size=BATCH_SIZE, seed=1
)

# Pick out an example
test_data = next(test_generator)
(masked, mask), original_image = test_data

# Show side by side
for i in range(len(original_image)):
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].imshow(masked[i, :, :, :])
    axes[1].imshow(mask[i, :, :, :] * 1.0)
    axes[2].imshow(original_image[i, :, :, :])
    plt.show()


def plot_callback(model):
    """Called at the end of each epoch, displaying our previous test images,
    as well as their masked predictions and saving them to disk"""

    # Get samples & Display them
    pred_img = model.predict([masked, mask])
    pred_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Clear current output and display test images
    for i in range(len(original_image)):
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].imshow(masked[i, :, :, :])
        axes[1].imshow(pred_img[i, :, :, :] * 1.0)
        axes[2].imshow(original_image[i, :, :, :])
        axes[0].set_title("Masked Image")
        axes[1].set_title("Predicted Image")
        axes[2].set_title("Original Image")

        plt.savefig(
            os.path.join("data", "test_samples", "img_{}_{}.png".format(i, pred_time))
        )
        plt.close()


# Instantiate the model
model = PConvUnet(weight_filepath=os.path.join("data", "logs"))

# Run training for certain amount of epochs
model.fit(
    train_generator,
    steps_per_epoch=1000,
    validation_data=val_generator,
    validation_steps=100,
    epochs=10,
    plot_callback=plot_callback,
    callbacks=[
        TensorBoard(
            log_dir=os.path.join("data", "logs", "initial_training"), write_graph=False
        )
    ],
)

n = 0
for (masked, mask), original_image in tqdm(test_generator):

    # Run predictions for this batch of images
    pred_img = model.predict([masked, mask])
    pred_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Clear current output and display test images
    for i in range(len(original_image)):
        _, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(masked[i, :, :, :])
        axes[1].imshow(pred_img[i, :, :, :] * 1.0)
        axes[0].set_title("Masked Image")
        axes[1].set_title("Predicted Image")
        axes[0].xaxis.set_major_formatter(NullFormatter())
        axes[0].yaxis.set_major_formatter(NullFormatter())
        axes[1].xaxis.set_major_formatter(NullFormatter())
        axes[1].yaxis.set_major_formatter(NullFormatter())

        plt.savefig(
            os.path.join("data", "test_samples", "img_{}_{}.png".format(i, pred_time))
        )
        plt.close()
        n += 1

    # Only create predictions for about 100 images
    if n > 100:
        break
