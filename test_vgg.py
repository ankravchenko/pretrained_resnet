#python 3.5, Tensorflow 2

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os
from os import path


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
mnist = tf.keras.datasets.mnist

from copy import copy

#from __future__ import absolute_import, division, print_function, unicode_literals

import random
from datetime import datetime
from tensorflow import keras

from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D, MaxPooling2D
import nltk


from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import VGG16, ResNet50
import numpy as np
import argparse

import logging

from random import randrange
import sys
import numpy
import matplotlib.pyplot as plt
numpy.set_printoptions(threshold=sys.maxsize)



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())



# load the input image using the Keras helper utility while ensuring
# that the image is resized to 224x224 pxiels, the required input
# dimensions for the network -- then convert the PIL image to a
# NumPy array
print("[INFO] loading and preprocessing image...")
image = image_utils.load_img(args["image"], target_size=(224, 224))
image = image_utils.img_to_array(image)


# our image is now represented by a NumPy array of shape (224, 224, 3),
# assuming TensorFlow "channels last" ordering of course, but we need
# to expand the dimensions to be (1, 3, 224, 224) so we can pass it
# through the network -- we'll also preprocess the image by subtracting
# the mean RGB pixel intensity from the ImageNet dataset
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)

# load the VGG16 network pre-trained on the ImageNet dataset
print("[INFO] loading network...")
model = ResNet50(weights="imagenet")
# classify the image
print("[INFO] classifying image...")
preds = model.predict(image)
P = decode_predictions(preds)

# loop over the predictions and display the rank-5 predictions +
# probabilities to our terminal
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
	print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
# load the image via OpenCV, draw the top prediction on the image,
# and display the image to our screen
(imagenetID, label, prob) = P[0][0]

print(label)
print(prob)

