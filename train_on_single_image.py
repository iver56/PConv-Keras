import gc
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from keras_pconv.pconv_model import PConvUnet
from keras_pconv.util import random_mask

# Settings
MAX_BATCH_SIZE = 128
BATCH_SIZE = 1  # can be increased to 4 if there is enough GPU memory


class DataGenerator(ImageDataGenerator):
    def flow(self, x, *args, **kwargs):
        while True:

            # Get augmented image samples
            original_image = next(super().flow(x, *args, **kwargs))

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

            # Yield ([original_image, mask],  original_image) training batches
            # print(masked.shape, original_image.shape)
            gc.collect()
            yield [masked, mask], original_image


if __name__ == "__main__":
    # Load image
    img = cv2.imread("./data/sample_image.jpg")
    assert img is not None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512)) / 255
    shape = img.shape
    print(f"Shape of image is: {shape}")

    # Load mask
    mask = random_mask(shape[0], shape[1])

    # Image + mask
    masked_img = deepcopy(img)
    masked_img[mask == 0] = 1

    # Show side by side
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].imshow(img)
    axes[1].imshow(mask * 255)
    axes[2].imshow(masked_img)
    plt.show()

    datagen = DataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )

    # Create generator from numpy arrays
    batch = np.stack([img for _ in range(MAX_BATCH_SIZE)], axis=0)
    generator = datagen.flow(x=batch, batch_size=BATCH_SIZE)

    # Get samples & Display them
    (masked, mask), original_image = next(generator)

    # Show side by side
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].imshow(masked[0, :, :, :])
    axes[1].imshow(mask[0, :, :, :] * 255)
    axes[2].imshow(original_image[0, :, :, :])

    def plot_callback(model):
        """Called at the end of each epoch, displaying our previous test images,
        as well as their masked predictions and saving them to disk"""

        # Get samples & Display them
        pred_img = model.predict([masked, mask])

        # Clear current output and display test images
        for i in range(len(original_image)):
            _, axes = plt.subplots(1, 3, figsize=(20, 5))
            axes[0].imshow(masked[i, :, :, :])
            axes[1].imshow(pred_img[i, :, :, :] * 1.0)
            axes[2].imshow(original_image[i, :, :, :])
            axes[0].set_title("Masked Image")
            axes[1].set_title("Predicted Image")
            axes[2].set_title("Original Image")
            plt.show()

    model = PConvUnet()
    model.fit(generator, steps_per_epoch=200, epochs=5, plot_callback=plot_callback)
