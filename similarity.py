from skimage.metrics import structural_similarity
import logging


def get_ssim_similarity(block1, block2):
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
        channel_axis = None
    else:
        mul = True
        channel_axis = 2

    # Calculate SSIM between the two images.
    sim, diff = structural_similarity(block1, block2, full=True, multichannel=mul, channel_axis=channel_axis)
    return round(sim, 4)

   
def get_similarity_with_neighbours(row, col, blocks, neighbours_index, how='ssim'):
    target_block = blocks[row][col].get_block_data() 
    
    if how == "ssim":
        sim = {"center": f"{row}-{col}"}
        for i, j in neighbours_index:
            sim[f"{i}-{j}"] = get_ssim_similarity(target_block, blocks[i][j].get_block_data())

    return sim


def display_similarities(sim_dict):
    """For printing the sorted list of similarities with its neighbours for a SINGLE block.

    Args:
        sim_dict (dict): Returned from get_similarity_with_neighbours(), 
                        contains the centre block and its neighbours along 
                        with their similarity score.
    """
    target = sim_dict['center']
    logging.info(f"Target block index {target}")
    scores = {block: sim_dict[block] for block in sim_dict if block != "center"}
    for k, v in sorted(scores.items(), key=lambda x : x[1], reverse=True):
        print(f"{k}: {v}")
