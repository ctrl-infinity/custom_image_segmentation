"""Entrypoint file for the project from which the tool can be used."""
from ImageBlocks import split_image_into_blocks
from pathlib import Path
import logging
import cv2

logging.basicConfig(level=logging.DEBUG)


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
