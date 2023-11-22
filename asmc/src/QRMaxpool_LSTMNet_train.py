import keras
import keras.backend as K
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Lambda, Input, Concatenate, LSTM, LeakyReLU, Reshape, Add

import tensorflow_addons as tfa
import tensorflow as tf

import os
import numpy as np
from PIL import Image
import timeit,time
import math
import pandas as pd

# ----------------------- QRMaxpool_LSTMNet ---------------------------
####################### EXPLANATION #########################
# We take the LER prediction from edgenet and the error prediction from 2
# of our NormCP models (NCPMaxPoolNet and NCPLSTMNet) and try to fit this to Quantile estimates using
# a neural net.
# EDGENET LER prediction should be the first element in train dataset
##############################################################

def Model_Builder(miscoverage: float):
    input1 = Input(shape=(3,))
    
    llayer1 = Dense(6)(input1)
    llayer1 = LeakyReLU()(llayer1)
    
    rlayer1 = Dense(6)(input1)
    rlayer1 = LeakyReLU()(rlayer1)
    
    llayer2 = Dense(1)(llayer1)
    llayer2 = LeakyReLU()(llayer2)
    
    rlayer2 = Dense(1)(rlayer1)
    rlayer2 = LeakyReLU()(rlayer2)
    
    branch = Lambda(lambda x: x[:,0])(input1)
    
    lout = Add()([llayer2, branch])
    
    rout = Add()([rlayer2, branch])
    
    model = Model(inputs=input1, outputs=[lout, rout])
    model.summary()

    model.compile('adam', loss=[tfa.losses.PinballLoss(tau=miscoverage / 2),
                                tfa.losses.PinballLoss(tau=1 - (miscoverage / 2))])
    return model

path = '/scratch/user/nini16/SEM/'
path_asmc = '/scratch/user/nini16/ASMC_SEM/'


X_train_path = path_asmc + "dataset/" + "X_train_QR_train3.npy"
y_train_path = path_asmc + "dataset/" + "y_train_QR_train2.npy"


Xtrain = np.load(X_train_path)
ytrain = np.load(y_train_path)

miscoverage_rate = 0.05
coverage_rate_string = "95"

print("**************************Now training left edge************************************")
left_model = Model_Builder()

callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=5, min_lr=0.0001),
    keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=3,
        )
]

history = left_model.fit(Xtrain[:,:3], [ytrain[:,0],ytrain[:,0]],
             batch_size=18,
             epochs=3,
             shuffle=True,
			 callbacks=callbacks)
			 
left_model.save(path_asmc + 'models/' + f'QRMaxpool_LSTMNet_ledge_{coverage_rate_string}.h5')

print("**************************Now training rigt edge************************************")
right_model = Model_Builder()

history = right_model.fit(Xtrain[:,3:], [ytrain[:,1],ytrain[:,1]],
             batch_size=18,
             epochs=3,
             shuffle=True,
			 callbacks=callbacks)

right_model.save(path_asmc + 'models/' + f'QRMaxpool_LSTMNet_redge_{coverage_rate_string}.h5')
