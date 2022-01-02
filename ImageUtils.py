import random
import numpy as np

""" This script implements the functions for data augmentation and preprocessing.
"""

def parse_record(record, training):
    """ Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    return image

def preprocess_image(image, training):
    """ Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3].
    """
    if training:
        ### YOUR CODE HERE
        # Resize the image to add four extra pixels on each side.
        image = np.pad(image, [(4, ), (4, ), (0, )], mode='constant')
        ### YOUR CODE HERE

        ### YOUR CODE HERE
        # Randomly crop a [32, 32] section of the image.
        # HINT: randomly generate the upper left point of the image
        random_row_1 = random.randint(1, 8)
        random_column_1 = random.randint(1, 8)
        random_row_2 = random_row_1 + 32
        random_column_2 = random_column_1 + 32

        cropped_image = np.zeros((32,32,3))
        for i in range(image.shape[2]):
            cropped_image[:,:,i] = image[random_row_1: random_row_2,random_column_1: random_column_2,i]

        ### YOUR CODE HERE
        # Randomly flip the image horizontally.
        x = random.randint(0,1)
        if(x == 1): cropped_image = np.flip(cropped_image, 1)

        image = cropped_image
        ### YOUR CODE HERE

    ### YOUR CODE HERE
    # Subtract off the mean and divide by the standard deviation of the pixels.
    norm_image = np.zeros((32,32,3))
    for i in range(image.shape[2]):
        sub_image = image[:,:,i]
        norm_image[:,:,i] = (sub_image - np.mean(sub_image))/np.std(sub_image)

    image = norm_image
    ### YOUR CODE HERE

    return image
