# import the modules we need

import os

# this is to make sure we don't get the tensorflow message
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# import the modules we need
import tensorflow as tf
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

'''
In Transformer, we don't have images as whole rather they first needs to be converted 
into patches.
this module patchify helps us just do that
'''
from patchify import patchify 

""" 
Hyperparameters 
hp is an empty dictionary
"""
hp = {}

'''
 Image size is Height x Width x Channels
 So we have
 200x200x3
 '''
hp["image_size"] = 200

#its an rgb image rgb
hp["num_channels"] = 3

"""
Patch size: Ph x Pw
Ph = 25
Pw = 25
Patch Size: 25 x 25
we need to make sure that the patch size is a multiple of image_size
"""
hp["patch_size"] = 25

""" Number of Patches (N)
num_patches = (200/25) * (200/25) = 8 * 8 = 64
simpler form: 
num_patches = (200 x 200) / (25 x 25)
"""
hp["num_patches"] = (hp["image_size"] ** 2) // (hp["patch_size"] ** 2)


"""
Now that we know the number of patches and also the path size,
we can define the shape of the flattened patches
We need to flatten the patches to feed them to the transformer
Transformed input
(64, 25 x 25 x 3)
= (256, 1875)

"""
# Define the shape of the patches that are extracted from the images
hp["flat_patches_shape"] = (hp["num_patches"], hp["patch_size"]*hp["patch_size"]*hp["num_channels"])

# Define the batch size
hp["batch_size"] = 32

# Define the number of epochs
hp["epochs"] = 500

# Define the learning rate
hp['lr'] = 1e-4

# Define number of classes
hp['num-classes'] = 5

# Define the name of the classes
hp['class-names'] = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# This function will help us create blank folders
def create_dir(path):
    """create directory"""
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path, split=0.1):
    #we need to split the data into test and training dataset
    # first we have the path, * because we have 5 different folders and need all of them
    # .jpg specifies that we need all the files with this type in those folders
    # we also shuffle the images.
    images = shuffle(glob(os.path.join(path, '**', '*.jpg')))
    # print(images)
    split_size = int(len(images) * split)
    # print(split_size)
    # split the data
    # we have not training, validation and testing
    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)

    return train_x, valid_x, test_x


if __name__ == "__main__":
    """seeding"""
    np.random.seed(42)
    tf.random.set_seed(42)
    
    """ Directory for storing files"""
    create_dir("files")

    """ Paths """
    dataset_path = "flower_photos"
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "log.csv")
    print(dataset_path)

    """ Dataset """
    train_x, valid_x, test_x = load_data(dataset_path)
    #print to see how many samples do we have
    print("Number of training samples: ", len(train_x))
    print("Number of validation samples: ", len(valid_x))
    print("Number of testing samples: ", len(test_x))
