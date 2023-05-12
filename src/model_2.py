## CNN for RANK Classification

import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import SGD, Adam

optimizer_type = 'adam'  # optimisation algorithm: SGD stochastic gradient decent 
loss = 'categorical_crossentropy'  # loss (cost) function to be minimised by the optimiser
metrics = ['categorical_accuracy']  # network accuracy metric to be determined after each epoch
num_hidden_nodes = 256  # number of nodes in hidden fully connected layer
dropout_ratio=0.25

input_shape = (28, 28, 1)
inputs = Input(shape=input_shape)

down_01 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same')(inputs)
down_01 = Activation('relu')(down_01)
down_01 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same')(down_01)
down_01 = Activation('relu')(down_01)
down_01_pool = MaxPooling2D((2, 2))(down_01)  
down_02 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')(down_01_pool)
down_02 = Activation('relu')(down_02)
down_02_pool = MaxPooling2D((2, 2))(down_02)  
down_03 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')(down_02_pool)
down_03 = Activation('relu')(down_03)

flatten = Flatten()(down_03)  

dense_01 = Dense(num_hidden_nodes)(flatten)
dense_01 = Activation('sigmoid')(dense_01)
dense_01 = Dropout(dropout_ratio)(dense_01)
dense_02 = Dense(13)(dense_01)
outputs = Activation('softmax')(dense_02)

model_2 = Model(inputs=inputs, outputs=outputs)
model_2.compile(optimizer=optimizer_type, loss=loss, metrics=metrics)