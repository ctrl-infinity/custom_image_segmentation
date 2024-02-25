import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG)


class Block:
    def __init__(self, name, data, color=None):
        self.name = name
        self.data = data
        self.color = color
        self.shape = data.shape
        self.prev_sim = 0

    def __str__(self):
        return f"{self.name}"
    
    def get_block_data(self):
        return self.data

    def set_new_color(self, new_color):
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

def get_block_neighbours(row, col, blocks):
    h, w = blocks.shape[0], blocks.shape[1]
    
    # Check if the block is present
    if not all([0 <= row < h, 0 <= col < w]):
        print("Invalid coordinates!")
        return -1

    # Getting the data(numpy array) of the block whose neighbours have to be found
    centre = blocks[row][col].get_block_data()
    
    pos_index = [-1, 0, 1]
    
    # Calculate all possible index of the neighbouring blocks 
    all_neighbours_index = [(row-i, col-j) for i in pos_index for j in pos_index if (i, j) != (0, 0)]
    
    # Filter out all the valid indices
    valid_neighbours_index = [(i, j) for i, j in all_neighbours_index if all([0 <= i < h, 0 <= j < w])]
    
    return valid_neighbours_index


def split_image_into_blocks(image, block_dim, draw_grid=True, to_gray=False):
    """
    Splits an image into the blocks of the given block dimensions. 
    If the `draw_grid` parameter is set to True, the method return an image with the gridlines drawn.
    If the `to_gray` parameter is set to True, the method would change the image file to grascale.

    Args:
        image (str): The image object for which the blocks need to be created.
        block_dim (tuple): The (height, width) of a block. The dimensions need to be proper factors of the dimensions of the image.
        draw_grid (bool, optional): A boolean indicating whether to get the image with a grid of blocks drawn over it. Defaults to True.
        to_gray (bool, optional): A boolean indicating whether to convert the image to grayscale. Defaults to False.

    Returns:
        Open CV Image Object: A CV object represnting the image.
        numpy.ndarray: A numpy array representing the blocks of the image row and col wise.

    Raises:
        Exception: An exception is raised if the image file is not found or if there is an error loading the image.

    Example:
        import cv2
        import logging

        # Set up logging
        logging.basicConfig(level=logging.INFO)

        # Read an image from file
        img = read_image("path/to/image.jpg") # Image size (800, 600, 3) for example

        # Split the image into blocks
        image_blocks, img = split_image_into_blocks(img, block_dim=(50, 50))
    """
    height, width = image.shape[0:2]
    block_height, block_width = block_dim
    
    # Check if the block dimensions are valid for the image
    if width % block_width != 0 and height % block_height != 0:
        logging.info(f"Dimensions of the image: {image.shape}")
        logging.info(f"Input image cannot be divided in {block_height} X {block_width} blocks")
        return -1, -1
    
    # Conver the image to to_grayscale if required
    if to_gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Number of rows and cols after splitting the image.
    rows, cols = (height // block_height), (width // block_width)
    
    blocks = []
    
    # Splitting the image into blocks.
    # NOTE: y variable ---> vertical axis and x ----> horizontal axis
    for i in range(rows):
        y1, y2 = i * block_height, (i + 1) * block_height
        row = []

        for j in range(cols):
            x1, x2 = j * block_width, (j+1) * block_width

            # To draw the block lines and the index on each block.
            # Splitting the blocks from the original image using numpy slicing
            img_block = Block(name=f"{i}-{j}", data=image[y1: y2, x1: x2])
            row.append(img_block)

            if draw_grid:
                image = cv2.rectangle(image, (x1, y1), (x2, y2), thickness=2, color=(35, 255, 100))
                image = cv2.putText(image, text=f"{i}{j}",
                                  org=((x1+x2)//2, (y1+y2)//2),
                                  fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                  fontScale=0.75,
                                  color=(35, 255, 100))

        blocks.append(row)

    return np.array(blocks), image



def main():
    path = Path.cwd()/"dummy.jpg"
    blocks, img = split_image_into_blocks(str(path), block_size=(100, 100), gray=False, draw_lines=True)
    print(blocks[0][0].shape)


if __name__ == "__main__":
    main()
