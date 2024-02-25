from skimage.metrics import structural_similarity

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
    
    
    
def similarity_with_neighbours(row, col, blocks, neighbours_index, how='ssim'):
    target_block = blocks[row][col].get_block_data() 
    
    if how == "ssim":
        sim = {"center": f"Block_{row}_{col}"}
        for i, j in neighbours_index:
            sim[f"Block_{i}_{j}"] = get_ssim_similarity(target_block, blocks[i][j].get_block_data())

    return sim