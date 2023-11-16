import tensorflow as tf
import tensorflow_addons as tfa
# from tensorflow.keras.losses import Loss

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import numpy as np
import math
import timeit
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Reshape, Lambda
from keras.constraints import max_norm
from PIL import Image
import keras.backend as K
from multi_gpu import make_parallel
import time
from random import shuffle

path = '/scratch/user/nini16/SEM/'

# Format of ideal/defining parameters:
# [sigma, alpha, Xi, width, space, noise, LER_(L or R)]

# change these to match the path of your generated dataset
# See Quantile_Regressor_defining_parameters_dataset.py for reference
# Depending on how the data was generated you might need to take a slice
# of the dataset for each edge (e.g. X_train[:, :7], y_train[:, 0] for
# left edge)

X_val = np.load(path + "models/" + "ideal_X_val.npy")
y_val = np.load(path + "models/" + "ideal_y_val.npy")

X_train = np.load(path + "models/" + "ideal_X_train.npy")
y_train = np.load(path + "models/" + "ideal_y_train.npy")


model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(7,)))
model.add(Dense(32, activation='relu')) 
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()

adam = keras.optimizers.adam(lr=1e-3)

quantile = 0.05 # lower quantile for 90% confidence
model.compile(loss = tfa.losses.PinballLoss(tau=quantile),
              optimizer=adam)

callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, min_lr=0.0001),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        ),
    # keras.callbacks.ModelCheckpoint(
        # path + 'models/' + 'Weights/compressedweights.{epoch:02d}-{val_loss:.2f}.hdf5',
        # monitor='val_loss',
        # save_best_only=True,
        # verbose=1),
]
			  
history = model.fit(X_train[:, :7], y_train[:, 0],
             batch_size=16,
             epochs=100,
             validation_data=(X_val[:, :7], y_val[:, 0]),
             shuffle=True,
			 callbacks=callbacks)
			  
model.save(path + 'models/' +'ideal_lower_90.h5')