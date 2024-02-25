"""Entrypoint file for the project from which the tool can be used."""
from image_blocks import Block
from pathlib import Path
import logging
import cv2
import numpy as np
from similarity import similarity_with_neighbours, get_block_neighbours

logging.basicConfig(level=logging.DEBUG)

def display_similarities(sim_dict):
    """For printing the sorted list of similarities with its neighbours for a SINGLE block.

    Args:
        sim_dict (dict): Returned from similarity_with_neighbours(), 
                        contains the centre block and its neighbours along 
                        with their similarity score.
    """
    target = sim_dict['center']
    logging.info(f"Target block index {target}")
    scores = {block: sim_dict[block] for block in sim_dict if block != "center"}
    for k, v in sorted(scores.items(), key=lambda x : x[1], reverse=True):
        print(f"{k}: {v}")
    
        
def display_image(img):
    cv2.imshow("Input Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None
    

def read_image(image_path, show_image=False):
    """
    Reads an image from the given path and returns it as a numpy array. If the `show_image` parameter is set to True, the function will also display the image in a new window.

    Args:
        image_path (str): A string representing the path to the image file.
        show_image (bool, optional): A boolean indicating whether to display the image in a new window. Defaults to False.

    Returns:
        numpy.ndarray: A numpy array representing the image.

    Raises:
        Exception: An exception is raised if the image file is not found or if there is an error loading the image.

    Example:
        import cv2
        import logging

        # Set up logging
        logging.basicConfig(level=logging.INFO)

        # Read an image from file
        img = read_image("path/to/image.jpg")

        # Display the image
        read_image("path/to/image.jpg", show_image=True)
    """
    try:
        # Loading image from the given path
        img = cv2.imread(image_path)
        logging.info(f"Selected image is {image_path}")
        
        # Display image in a new window
        if show_image:
            display_image(img)
            
            
        logging.info(f"The dimensions of the image are {img.shape}")
        return img
            
    except Exception as e:
        print(f"Check your image file present in {image_path}. It gave the following error \n\t{e}")


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
    

def main(image_path):
    # Read the image
    img = read_image(image_path, show_image=False)
    
    # Split the image into blocks
    image_blocks, img = split_image_into_blocks(img, block_dim=(100, 100))
    
    # Calculate 
    for i in range(image_blocks.shape[0]):
        for j in range(image_blocks.shape[1]):
            neighbours = get_block_neighbours(i, j, image_blocks)
            # print(f" Neigbours for ({i}, {j}) are: {neighbours}")
    
    
    # Calculate similarity with the neighbours
    neighbours_0_0 = get_block_neighbours(4, 1, image_blocks)
    s = similarity_with_neighbours(4, 1, image_blocks, neighbours_0_0)
    
    # 
    display_similarities(s)
    
    # Display the image
    display_image(img)
    
    # Getting the individual blocks in array
    # And getting image for visualizing the blocks
    # img_blocks, img = split_image_into_blocks(str(image_path), block_size=(200, 200), gray=True, draw_lines=True)
    
    # if not isinstance(img_blocks, int):
    #     logging.info(f"Total number of blocks: {img_blocks.size}")
    
    # cv2.imshow("Input Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    # image_name = input("Enter the image name: ")
    image_name="bottle.jpg"
    image_path = str(Path.cwd()/"inputs"/image_name)
    main(image_path=image_path)
