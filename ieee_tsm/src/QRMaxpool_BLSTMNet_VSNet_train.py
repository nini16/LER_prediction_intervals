import keras
import keras.backend as K
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Lambda, Input, Concatenate, LSTM, LeakyReLU, Reshape, Add, Multiply

import tensorflow_addons as tfa

import os
import numpy as np
from PIL import Image
import timeit,time
import math
import pandas as pd

# ----------------------- QRMaxpool_BLSTMNet_VSNet_train ---------------------------
####################### EXPLANATION #########################
# We take the LER prediction from edgenet and the error prediction from 2
# of our NormCP models (NCPMaxPoolNet and NCPBLSTMNet) plus a fourth parameter which represents
# an estimate of the level of noise (standard deviation) of an input noisy image and try to fit this
# to Quantile estimates using a neural net.

# We use th tanh activation function on the multiplier
# lMultiplier = Dense(1, activation='tanh')(mult_branch)

# Note:The training dataset was pre-generated and stored as a ".npy" file
# EDGENET LER prediction should be the first element in train dataset and the noise level estimate
# is the last element.

# The mechanism for noise level estimation employed is a well-known wavelet thresholding scheme
# known as VisuShrink. 
# Implementation: https://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.estimate_sigma
##############################################################

def Model_Builder(miscoverage: float):
    input1 = Input(shape=(4,))
    
    main_branch = Lambda(lambda x: x[:,:3])(input1) # the 3 main inputs: (edgenet, NCPBLSTMNet, NCPMaxpool)
        
    llayer1 = Dense(6)(main_branch)
    llayer1 = LeakyReLU()(llayer1)
    
    rlayer1 = Dense(6)(main_branch)
    rlayer1 = LeakyReLU()(rlayer1)
    
    llayer2 = Dense(1)(llayer1)
    llayer2 = LeakyReLU()(llayer2)
    
    rlayer2 = Dense(1)(rlayer1)
    rlayer2 = LeakyReLU()(rlayer2)
    
    mult_branch = Lambda(lambda x: x[:,3:4])(input1) # noise paramaeter
    
    lMultiplier = Dense(1, activation='tanh')(mult_branch)
    rMultiplier = Dense(1, activation='tanh')(mult_branch)
    
    llayer3 = Multiply()([llayer2, lMultiplier])
    rlayer3 = Multiply()([rlayer2, rMultiplier])
    
    add_branch = Lambda(lambda x: x[:,0])(input1) # edgenet LER
    
    lout = Add()([llayer3, add_branch])
    
    rout = Add()([rlayer3, add_branch])
    
    model = Model(inputs=input1, outputs=[lout, rout])
    model.summary()

    model.compile('adam', loss=[tfa.losses.PinballLoss(tau=miscoverage / 2),
                                tfa.losses.PinballLoss(tau=1 - (miscoverage / 2))])
    return model

path_tsm = '/scratch/user/nini16/IEEE_TSM/'


X_train_path = path_tsm + "dataset/" + "X_train_QRMaxpool_BLSTMNet_VSNet_train.npy"
y_train_path = path_tsm + "dataset/" + "y_train_QR_train2.npy"


Xtrain = np.load(X_train_path)
ytrain = np.load(y_train_path)

miscoverage_rate = 0.05
coverage_rate_string = "95"

print("**************************Now training left edge************************************")
left_model = Model_Builder(miscoverage_rate)

callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=2, min_lr=0.0001),
    keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=2,
        )
]

history = left_model.fit(Xtrain[:,[0,1,2,6]], [ytrain[:,0],ytrain[:,0]],
             batch_size=18,
             epochs=5,
             shuffle=True,
			 callbacks=callbacks)
			 
left_model.save(path_tsm + 'models/' + f'QRMaxpool_BLSTMNet_VSNet_ledge_{coverage_rate_string}.h5')

print("**************************Now training rigt edge************************************")
right_model = Model_Builder(miscoverage_rate)

history = right_model.fit(Xtrain[:,[3,4,5,7]], [ytrain[:,1],ytrain[:,1]],
             batch_size=18,
             epochs=5,
             shuffle=True,
			 callbacks=callbacks)

right_model.save(path_tsm + 'models/' + f'QRMaxpool_BLSTMNet_VSNet_redge_{coverage_rate_string}.h5')
