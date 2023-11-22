import tensorflow as tf

import numpy as np
import math
import timeit

import keras
from keras.models import Sequential,load_model,Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Conv2DTranspose, Reshape, MaxPooling2D, BatchNormalization, UpSampling2D, Input, Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.constraints import max_norm
from PIL import Image
import keras.backend as K
from multi_gpu import make_parallel
import time
from random import shuffle

start = time.time()

path = '/scratch/user/nini16/SEM/'

#getting the data

num_validation = 2880
num_test = 8640

sigmas = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]	

widths = [20, 30]
noises = [2, 3, 4, 5, 10, 20, 30, 50, 100, 200]

input = Input(shape=(1024, 64, 1))
y = Conv2D(64, (3, 3), padding='same',input_shape= (1024,64,1), strides=(2, 2), activation = 'relu')(input)
y = BatchNormalization(axis=3)(y)
y = Dropout(0.1)(y)

y = Conv2D(128, (3, 3), padding='same', strides=(2, 2), activation = 'relu')(y)
y = BatchNormalization(axis=3)(y)
y = Dropout(0.1)(y)

y = Conv2D(256, (3, 3), padding='same', strides=(2, 2), activation = 'relu')(y)
y = BatchNormalization(axis=3)(y)
y = Dropout(0.1)(y)

y = GlobalMaxPooling2D()(y)

llayer2 = Dense(64, activation = 'relu')(y)
rlayer2 = Dense(64, activation = 'relu')(y)
llayer3 = Dense(16, activation = 'relu')(llayer2)
rlayer3 = Dense(16, activation = 'relu')(rlayer2)
llayer4 = Dense(4, activation = 'relu')(llayer3)
rlayer4 = Dense(4, activation = 'relu')(rlayer3)
lout = Dense(1, activation = 'relu')(llayer4)
rout = Dense(1, activation = 'relu')(rlayer4)
out = Concatenate()([lout, rout])

model = Model(inputs=input, outputs=out)

model.summary()

G = 1
if G > 1:
   model = make_parallel(model,G)


adam = keras.optimizers.adam(lr=1e-3)

model.compile(loss = 'mean_absolute_error',
              optimizer=adam)


# ----------------------------------load training data and train on it --------------------------------------- 

num_training = 9920*9

noises = [2, 3, 4, 5, 10, 20, 30, 50, 100, 200]

Xis = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
Xis.remove(10)			# remove 10, 20, 30 and 40 value (This value will be used to create validation and test set)
Xis.remove(20)
Xis.remove(30)	
Xis.remove(40)


X_train = np.load(path + "dataset/" + "NoiseCompression_Xtrain.npy")
y_train = np.load(path + "dataset/" + "compressed_y_train.npy")


print('Train data shape: ', X_train.shape)


callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=3, min_lr=0.0001),
    keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=4,
        ),
    # keras.callbacks.ModelCheckpoint(
        # path + 'Weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        # monitor='val_acc',
        # save_best_only=True,
        # verbose=1),
]
	
history = model.fit(X_train, y_train,
             batch_size=18,
             epochs=13,
             shuffle=True,
			 callbacks=callbacks)
			  

model.save(path + 'models/' +'NCPMaxpool.h5')
del model  # deletes the existing model


print("Execttion Time= ", time.time() - start)


