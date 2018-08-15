#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 15:31:59 2018

@author: hitesh
"""

#Importing the required Libraries
from __future__ import print_function 
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D as Conv2D
from keras.layers import MaxPooling2D as Max2D
from keras import backend as K
from keras.datasets import mnist 
from sklearn.model_selection import train_test_split

# Input image dimensions 
img_row , img_col = 28 , 28
input_img = (img_row, img_col , 1)

# Splitting data set in to training and test data 
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], img_row, img_col, 1)
X_test = X_test.reshape(X_test.shape[0], img_row, img_col, 1)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

#Rescaling the images of values between [0,1]
X_train = X_train/255.0
X_test = X_test/ 255.0

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 26)
y_test = keras.utils.to_categorical(y_test, 26)

# Construting the CNN Architecture 

model = Sequential()

model.add(Conv2D(32,kernel_size = (3,3) , activation = 'relu', 
                 input_shape = input_img))

model.add(Conv2D(64,kernel_size = (3,3), activation = 'relu'))

model.add(Max2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128,activation = 'relu'))

model.add(Dropout(0.25))

model.add(Dense(26,activation = 'softmax'))

# Complie the model
model.compile(loss =keras.losses.categorical_crossentropy, 
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Model variables
batch_size = 128
num_classes = 26
epochs = 10

# Train the model
model.fit(X_train, y_train, batch_size= batch_size,
          epochs= epochs,verbose = 1, validation_data=(X_test,y_test))

# Save the model for further reference
model.save("emnist_model_cnn.h5")

# Evaluate the model using Accuracy and Loss
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

            
# Alphabet recognition code comes here ------
























