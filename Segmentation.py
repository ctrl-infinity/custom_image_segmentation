import cv2

from ImageBlocks import blocks_similarity
from random import choice
from PIL.ImageColor import getcolor
import numpy as np


def change_shade(color, change_by, weight=0.15):
    """
    Change shade of any color
    :param color: tuple
                RGB or BGR
    :param change_by: int
                Number of levels down/up in the shade of the color.
    :param weight: float
                Multiplied by the change_by and then added to the original color component.
    :return: tuple
                Returns the new color.
    """
    new_col = []
    for c in color:
        # Changing the components
        c = c * (1 + (change_by * weight))

        # Make sure that the new component remains below 255
        if c > 255:
            c = 255

        new_col.append(int(c))

    return tuple(new_col)


def random_color():
    """
    Creates a random color using hexadecimal digits. And then converts it into RGB code.
    :return:
    """
    s = [*range(10), *list("ABCDEF")]
    # Generate color in hex code.
    col = "#" + "".join(str(choice(s)) for _ in range(6))

    return col
    # Convert to BGR
    # bgr_color = getcolor(col, "RGB")[::-1]
    #
    # return bgr_color


def image_segmentation(blocks, color_shading=False, thresh=0.9):
    """
    Performs segmentation on the blocks given as input.
    :param blocks: numpy array
                An array of subparts of an image
    :param color_shading: Boolean
                If color shading is True, then the program uses various shades of colors.
    :param thresh: float, optional
                Threshold value for comparision with similarity scores.
    :return:
    """
    # Loop through the blocks array
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            # Assign a color if the block in question has no color.
            if blocks[i][j].color is None:
                blocks[i][j].change_color(random_color())

            # Calculate similarity with neighbours
            neighbours = similarity_with_neighbours(i, j, blocks)

            param = 0.7 if color_shading else thresh

            for n in neighbours:
                x, y = list(map(int, n.split("-")))

                if (neighbours[n] >= param) and (blocks[x][y].color is None):
                    change = int(neighbours[n] * 10) - int(10 * thresh) if color_shading else 0
                    new_col = change_shade(blocks[i][j].color, change_by=change)
                    blocks[x][y].change_color(new_col)
                    blocks[x][y].prev_sim = neighbours[n]

                elif (neighbours[n] >= thresh) and (blocks[x][y].color is not None):
                    if neighbours[n] >= blocks[x][y].prev_sim:
                        change = int(neighbours[n] * 10) - int(10 * thresh) if color_shading else 0
                        new_col = change_shade(blocks[i][j].color, change_by=change)
                        blocks[x][y].change_color(new_col)

    return blocks


def similarity_with_neighbours(row, col, blocks):
    """
    Calculates similarity of a block ij with its neighbors.
    :param row: int
            Row number. Indexing starting from 0.
    :param col: int
            Column number. Indexing starting from 0.
    :param blocks: Set of all the blocks
    :return: dict
            keys: neighbor names, value: similarity score
    """
    # Check for valid row and col values.
    h, w = blocks.shape[0], blocks.shape[1]
    if not all([0 <= row < h, 0 <= col < w]):
        print("Invalid coordinates!")
        return -1

    # Identify the centre point.
    centre = blocks[row][col]
    pos_index = [-1, 0, 1]

    # Calculate the indices of its neighbors.
    neighbours_index = [(row-i, col-j) for i in pos_index for j in pos_index if (i, j) != (0, 0)]

    # If the neighbor index is present in our array of blocks
    bool_index = [(i, j) for i, j in neighbours_index if all([0 <= i < h, 0 <= j < w])]
    sim = {}
    for i, j in bool_index:
        sim[f"{i}-{j}"] = blocks_similarity(centre.data, blocks[i][j].data)

    return sim


def form_images(blocks, change_color=False):
    """
    This method reads a 2D array of various image blocks of same size and then forms an image using them.
    :param blocks: numpy ndarray
                Array of blocks which will combine and form an image.
    :param change_color: bool (optional)
                If you want the color to be upgraded, then True.
    :return: numpy array
            Formed Image
    """
    # Check if the color needs to be updated.
    if change_color:
        for row in range(len(blocks)):
            for col in range(len(blocks[row])):
                blocks[row][col].print_color()

    # Concatenating the blocks to each other and then concatenating the rows of blocks
    img = np.concatenate([np.concatenate(([block.data for block in row]), axis=1) for row in blocks], axis=0)

    return img


def overlapping_output(original_image, output, weight1=0.5, weight2=0.5):
    out = cv2.addWeighted(original_image, weight1, output, weight2, 0)
    return out



