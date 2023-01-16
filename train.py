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

"""
process individual image path and extract lables from each class
"""
def process_image_label(path):
    """ Reading Images """
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    # resize the image
    image = cv2.resize(image, (hp["image_size"], hp["image_size"]))
    # normalize all the pixel values
    # image = image / 255.0
    print(image.shape)

    """ Preprocessing to patches """
    patch_shape = (hp['patch_size'], hp["patch_size"], hp["num_channels"])
    # now we use the patchify function
    # first input is the image(h,w,c_num) then patch_shape the next is the step 
    # step is the size of the patch, so if 0 to 25 the next would be the 26th item
    patches = patchify(image, patch_shape, step=hp["patch_size"])
    # after printing the patches and finding out the shape of it which was
    # (8, 8, 25, 25, 3) we then manually use np.reshape to make 8 x 8 = 64 
    # after that we use a for loop to save these 64 patches
    patches = np.reshape(patches, (64, 25, 25, 3))

    # This is just a test to see the patches in files folder
    # and see how it works
    # We comment it out
    
    # for i in range(64):
    #     cv2.imwrite(f"files/{i}.png", patches[i])
    # print(patches.shape)

    # flatten the patches into an apropriate shape (the hyper parameter)
    patches = np.reshape(patches, hp["flat_patches_shape"])
    # now we need to provide them a datatype    
    patches = patches.astype(np.float32)

    """ Label """
    # we need to extract the label from the path and the name of the label
    # so we print the label path first to extract the label from it
    # print(path)
    # now that we know the path items are seperated with /:
    # we split the path and extract the last item which is the label
    class_name = path.split("/")[-2]
    # print(class_name)
    
    

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

    # test the process_image_label function 
    print(process_image_label(train_x[3]))