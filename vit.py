
"""
 ***
    Implementation of VisionTransformer
 ***
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


# defining the VisionTransformer function
def ViT(cf):
    input_shape = (cf["num_patches"], cf["patch_size"] * cf["patch_size"] * cf["num_channels"]) 
    inputs = Input(shape=input_shape) # (None, 256, 3072)
    # print(inputs.shape) 

    """ Patch + Embedding """
    patch_embed = Dense(cf["hidden_dim"])(inputs) # the shape of this layer print(patch_embed.shape) # (None, 256, 768)
    positions = tf.range(start=0,limit=cf["num_patches"], delta=1)
    pos_embed = Embedding(input_dim=cf["num_patches"], output_dim=cf["hidden_dim"])(positions) ## (256, 768)
   
    embed = patch_embed + pos_embed ## (None, 256, 768)
    




# configuration
if __name__ == '__main__':
    config = {}
    config["num_layers"] = 12
    config["hidden_dim"] = 768
    config["num_heads"] = 12
    config["num_dim"] = 3072
    config["dropout_rate"] = 0.1
    config["num_patches"] = 256
    config["patch_size"] = 32
    config["num_channels"] = 3


ViT(config)