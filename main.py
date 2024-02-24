"""Entrypoint file for the project from which the tool can be used."""
from ImageBlocks import split_image_into_blocks
from pathlib import Path
import logging
import cv2

logging.basicConfig(level=logging.DEBUG)


def read_image(image_path, show_image=False):
    try:
        # Loading image from the given path
        img = cv2.imread(image_path)
        logging.info(f"Selected image is {image_path}")
        
    # Display image in a new window
        if show_image:
            cv2.imshow("Input Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        logging.info(f"The dimensions of the image are {img.shape}")
        return img
            
    except Exception as e:
        print(f"Check your image file present in {image_path}. It gave the following error \n\t{e}")
    

    

def main(image_path):
    # Read the image
    img = read_image(image_path, True)
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
