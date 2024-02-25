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


def main():
    path = Path.cwd()/"dummy.jpg"
    blocks, img = split_image_into_blocks(str(path), block_size=(100, 100), gray=False, draw_lines=True)
    print(blocks[0][0].shape)


if __name__ == "__main__":
    main()
