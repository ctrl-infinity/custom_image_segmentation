"""Entrypoint file for the project from which the tool can be used."""
from image_blocks import Block, split_image_into_blocks, get_block_neighbours
from pathlib import Path
import logging
import cv2
import numpy as np
from similarity import get_similarity_with_neighbours, get_ssim_similarity, display_similarities
from image_utils import read_image, display_image
from Segmentation import image_segmentation, random_color, change_shade, form_images

logging.basicConfig(level=logging.DEBUG)

    
def main(image_path):
    # Read the image
    img = read_image(image_path, show_image=False)
    
    # Split the image into blocks
    image_blocks, img = split_image_into_blocks(img, block_dim=(10, 10))
    
    # # Calculate 
    # for i in range(image_blocks.shape[0]):
    #     for j in range(image_blocks.shape[1]):
    #         neighbours = get_block_neighbours(i, j, image_blocks)
    #         # print(f" Neigbours for ({i}, {j}) are: {neighbours}")
    
    
    # # Calculate similarity with the neighbours
    # neighbours_0_0 = get_block_neighbours(4, 1, image_blocks)
    # s = get_similarity_with_neighbours(4, 1, image_blocks, neighbours_0_0)
    
    # # 
    # display_similarities(s)
    
    # # Display the image
    # display_image(img)
    
    # print(change_shade((153, 91, 210), change_by=4))
    # Using image segmentation function
    s = image_segmentation(image_blocks, color_shading=True, thresh=0.9)
    
    output = form_images(s, change_color=True)
    
    display_image(output)
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
