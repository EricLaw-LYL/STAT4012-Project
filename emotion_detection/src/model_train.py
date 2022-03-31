"""
Train a emotion dectection model using tensorflow, keras_vggface
"""

import tensorflow as tf
from keras import layers, Sequential
from keras_vggface import VGGFace

# Create the model
vgg_model = VGGFace(include_top = False, model = "vgg16", input_shape = (48, 48, 1))
vgg_model.add(layers.Flatten())

print(vgg_model.summary())
