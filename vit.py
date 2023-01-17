
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

def ViT(cf):
    input_shape = (cf["num_patches"], cf["patch_size"] * cf["patch_size"] * cf["num_channels"]) 

 
# configuration
if __name__ == '__main__':
    config = {}
    config["num_layers"] = 12
    config["hidden_layers"] = 768
    config["num_heads"] = 12
    config["num_dim"] = 3072
    config["dropout_rate"] = 0.1
    config["num_patches"] = 256
    config["patch_size"] = 32
    config["num_classes"] = 3


