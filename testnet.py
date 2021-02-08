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

#tensorboard logs
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

import logging



from random import randrange
import sys
import numpy
import matplotlib.pyplot as plt
numpy.set_printoptions(threshold=sys.maxsize)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()	

def load_data(): # use as a global variable maybe? less hassle
	#check for pickled and load pickled
	#load MNIST set
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()	
	#pickle
	return train_images, train_labels, test_images, test_labels
	
#generate1digit_dataset(n)
#generate2digit(n, digitlist)

def generate_cnn_model():
#create model
	model = tf.keras.models.Sequential([
		keras.layers.Conv2D(64, kernel_size=3, name='conv1', activation='relu', input_shape=(84,84,1)),
		keras.layers.AveragePooling2D(name='pool1'),
		keras.layers.Conv2D(32, kernel_size=3, name='conv2', activation='relu'),
		keras.layers.AveragePooling2D(name='pool2'),
		keras.layers.Conv2D(32, kernel_size=3, name='conv3', activation='relu'),
		keras.layers.AveragePooling2D(name='pool3'),
		keras.layers.Flatten(name='flatten'),
		keras.layers.Dense(10, activation='softmax', name='dense1')
		])

	#new_learning_rate = 0.0001 
	#opt = tf.optimizers.Adam(new_learning_rate)
	#opt = keras.optimizers.Adam(learning_rate=1e-4)
	model.compile(optimizer='adam',
		loss='sparse_categorical_crossentropy',
      	        metrics=['accuracy'])

	return model


model = generate_cnn_model()

