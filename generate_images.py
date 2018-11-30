import os
import math
import random

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

size = (512, 512)

fonts = {}


def get_font(font_size):
    if font_size not in fonts:
        fonts[font_size] = ImageFont.truetype("BebasNeue-Regular.ttf", font_size)
    return fonts[font_size]


if __name__ == "__main__":
    random.seed(2345)

    os.makedirs(os.path.join("data", "training"))
    os.makedirs(os.path.join("data", "validation"))
    os.makedirs(os.path.join("data", "test"))

    # Generate and store images
    for i in tqdm(range(1000)):

        image = Image.new("RGB", size, (255, 255, 255))

        # Drawing context
        ctx = ImageDraw.Draw(image)

        font_size = math.ceil(image.size[0] / 2)
        font = get_font(font_size)

        ctx.text(
            (size[0] / 2 - 0.6 * font_size, int(image.size[1] - 1.5 * font_size)),
            str(i),
            font=font,
            fill=(20, 20, 20, 20),
        )

        folder = "training"
        r = random.random()
        if r < 0.1:
            folder = "validation"
        elif r < 0.2:
            folder = "test"

        image.save(os.path.join("data", folder, "{0:04d}.png".format(i)))
