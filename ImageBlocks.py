import cv2
from skimage.metrics import structural_similarity
import numpy as np
from pathlib import Path


class Block:
    def __init__(self, name, data, color=None):
        self.name = name
        self.data = data
        self.color = color
        self.shape = data.shape
        self.prev_sim = 0

    def change_color(self, new_color):
        self.color = new_color

    def print_color(self):
        if self.color is not None:
            s = np.tile(self.color, (self.shape[0], 1))
            s = np.tile(s, (self.shape[1], 1))
            if len(self.shape) == 2:
                s = s.reshape((self.shape[0], self.shape[1], 3))
            elif len(self.shape) == 3:
                s = s.reshape(self.shape)
            print(s.shape)
            self.data = s / 255.0

    def print_data(self):
        print(self.data)

    def __str__(self):
        return f"{self.name}"


def split_image(path, block_size, gray=False, draw_img=False):
    """
    Takes an image as input and then splits the image into various blocks
    of the given block_size.
    :param path: str
                Path of the image that we want to split.
    :param block_size: iterable
                Size of each block. The dimensions should divide the
                dimensions of our input image so that we get blocks of
                equal sizes.
    :param gray: Boolean, optional
                To convert the image to grayscale before splitting it into
                blocks.
    :param draw_img: Boolean, optional
                To draw lines on the image in order to represent the blocks.
    :return: img_blocks, img: numpy array
                Image blocks consists of the blocks row-wise.
                Original image is also returned with block lines.
    """
    # Reading the image
    img = cv2.imread(path)
    height, width = img.shape[0:2]
    block_height, block_width = block_size

    # Checking if the block dimensions are valid.
    if width % block_width != 0 and height % block_height != 0:
        print(f"Original shape: {img.shape}")
        print(f"Input image cannot be divided in {block_height} X {block_width} blocks")
        return -1, -1

    blocks = []
    # To convert the image into grayscale.
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Number of rows and cols after splitting the image.
    rows, cols = (height // block_height), (width // block_width)

    # Splitting the image into blocks.
    for i in range(rows):
        y1, y2 = i * block_height, (i + 1) * block_height
        row = []

        for j in range(cols):
            x1, x2 = j * block_width, (j+1) * block_width

            # To draw the block lines and the index on each block.
            # Splitting the blocks from the original image using numpy slicing
            # img_block = img[y1: y2, x1: x2]
            img_block = Block(name=f"{i}-{j}", data=img[y1: y2, x1: x2])
            row.append(img_block)

            if draw_img:
                img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(35, 255, 100), thickness=2)
                img = cv2.putText(img, text=f"{i}{j}",
                                  org=((x1+x2)//2, (y1+y2)//2),
                                  fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                  fontScale=0.75,
                                  color=(35, 255, 100))

        blocks.append(row)

    return np.array(blocks), img


def blocks_similarity(block1, block2):
    """
    Calculates the similarity between two blocks using SSIM metric from Skimage.
    :param block1: numpy array
                A subset of an image array which represents a block.
    :param block2: numpy array
                A subset of an image array which represents a block.
    :return: sim: float
                Returns similarity score rounded to four significant figures.
    """
    # Check if the blocks are grayscale or not.
    if block1.shape[-1] == 1:
        mul = False
    else:
        mul = True

    # Calculate SSIM between the two images.
    sim, diff = structural_similarity(block1, block2, full=True, multichannel=mul)
    return round(sim, 4)


def main():
    path = Path.cwd()/"dummy.jpg"
    blocks, img = split_image(str(path), block_size=(100, 100), gray=False, draw_img=True)
    print(blocks[0][0].shape)


if __name__ == "__main__":
    main()
