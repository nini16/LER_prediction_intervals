import keras
import keras.backend as K
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Lambda, Input, Concatenate, LSTM, LeakyReLU, Reshape, Bidirectional

import tensorflow_addons as tfa
import tensorflow as tf

import os
import numpy as np
from PIL import Image
import timeit,time
import math
import pandas as pd


# ----------------------- NCPBLSTMNet ---------------------------
####################### EXPLANATION #########################
# In this approach we feed a Noise image into a CNN + BLSTM + FC network.
# We try to predict the taret error for each edge separately.

# Note that all Noise images were pre-extracted using NoiseImage_DataGen.py
# into the dataset NoiseCompression_Xtrain.npy
##############################################################

def Model_Builder():
    input1 = Input(shape=(1024, 64, 1))
    
    y = Conv2D(64, (3, 3), padding='same',input_shape= (1024,64,1), strides=(2, 1), activation = 'relu')(input1)
    y = BatchNormalization(axis=3)(y)
    y = Dropout(0.02)(y)
    
    y = Conv2D(128, (3, 3), padding='same', strides=(2, 1), activation = 'relu')(y)
    y = BatchNormalization(axis=3)(y)
    y = Dropout(0.02)(y)
    
    y = Conv2D(256, (3, 3), padding='same', strides=(2, 1), activation = 'relu')(y)
    y = BatchNormalization(axis=3)(y)
    y = Dropout(0.02)(y)
    
    y = Conv2D(64, (3, 3), padding='same',input_shape= (1024,64,1), strides=(2, 2), activation = 'relu')(y)
    y = BatchNormalization(axis=3)(y)
    
    y = MaxPooling2D((4,4))(y)
    y = BatchNormalization(axis=3)(y)
    
    y = Reshape((64,128))(y)
    
    y = Bidirectional(LSTM(64))(y)
    
    llayer2 = Dense(32)(y)
    llayer2 = LeakyReLU()(llayer2)
    
    llayer3 = Dense(8)(llayer2)
    llayer3 = LeakyReLU()(llayer3)
    
    llayer4 = Dense(4)(llayer3)
    llayer4 = LeakyReLU()(llayer4)
    
    lout = Dense(1, activation = 'relu')(llayer3)
    
    model = Model(inputs=input1, outputs=lout)
    model.summary()
    
    adam = keras.optimizers.adam(lr=5e-2)
    model.compile(loss = 'mean_absolute_error', optimizer=adam)
    return model


path = '/scratch/user/nini16/SEM/'
path_asmc = '/scratch/user/nini16/ASMC_SEM/'

noise_image_path = path + "dataset/" + "NoiseCompression_Xtrain.npy"
y_train_path = path + "dataset/" + "compressed_y_train.npy"


input1 = np.load(noise_image_path)
outputErr = np.load(y_train_path)
 
print("**************************Now training left edge************************************")
left_model = Model_Builder()

callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=5, min_lr=0.0001),
    keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=3,
        )
]

history = left_model.fit(input1[:], outputErr[:,0],
             batch_size=18,
             epochs=3,
             shuffle=True,
			 callbacks=callbacks)
			 
left_model.save(path_asmc + 'models/' +'NCPBLSTMNet_ledge.h5')

print("**************************Now training rigt edge************************************")
right_model = Model_Builder()

history = right_model.fit(input1[:], outputErr[:,1],
             batch_size=18,
             epochs=3,
             shuffle=True,
			 callbacks=callbacks)

right_model.save(path_asmc + 'models/' +'NCPBLSTMNet_redge.h5')
