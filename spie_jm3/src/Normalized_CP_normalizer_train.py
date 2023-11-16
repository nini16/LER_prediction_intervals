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
from keras.models import Sequential,load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Reshape, Lambda, Conv1D, MaxPooling1D
from keras.constraints import max_norm
from keras.regularizers import l2
from PIL import Image
import keras.backend as K
from multi_gpu import make_parallel
import time
from random import shuffle

start = time.time()

path = '/scratch/user/nini16/SEM/'

# Load AutoEncoder model
model = load_model(path + 'models/' + 'AutoEncoder3_norm_run_1_epoch_4.h5')
model = Model(inputs=model.input, outputs=model.get_layer('conv2d_7').output) # conv2d_6

EdgeNet = load_model(path + 'models/' + 'EDGEnet2_int_L1_epoch_4.h5')
EdgeNet.add(Lambda(lambda x: K.std(x*64, axis=1)/2))



# ---------------------------------- load validation data ---------------------------------------

num_validation = 2880
num_test = 8640

X_val = np.zeros((num_validation,64,1,1))
y_val = np.zeros((num_validation,1,2,1))



sigmas = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]	

widths = [20, 30]
noises = [2, 3, 4, 5, 10, 20, 30, 50, 100, 200]					

Xis = [20]
count = 0

for sigma in sigmas:
	for alpha in alphas:
		for Xi in Xis:
			for width in widths:
				for s in range(2):
					for noise in noises:
						space = math.floor(width*2**s)
						shift = math.floor(-25 + (width + space/2 + Xi + alpha*10 + sigma*10)%16) 
						
						noisy_file = path + 'images/' + 'noisy_images2/nim_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.tiff'
						linescan_file = path + 'images/' + 'linescans/linescan_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '.txt'
						
						linescan = []
						with open(linescan_file,'r') as f:
							for i,line in enumerate(f):
								if i < 3000:
									a, b = line.split(',')
									linescan.append(float(b))
								else:
									break
						linescan = linescan[:2048]
						leftline = np.array(linescan[:1024])
						rightline = linescan[1024:]
						rightline.reverse()
						rightline = np.array(rightline)
						leftline = leftline + shift
						rightline = rightline + shift
						
						# removed rounding
						lline_std = leftline.std()/2
						rline_std = rightline.std()/2
						
						imnoisy = np.array(Image.open(noisy_file))/256
						imnoisy = np.reshape(imnoisy, (1,1024,64,1))
						
						
						orig = np.array([[ [lline_std], [rline_std] ]])
						pred = EdgeNet.predict(imnoisy)
						
						target = -np.log(np.abs(orig - pred))
						
						X_val[count] = model.predict(imnoisy)
						y_val[count] = target
						count += 1					
print('Validation_count: ',count)

X_val = np.reshape(X_val,(num_validation,64))
y_val = np.reshape(y_val,(num_validation,2))

print('Validation data shape: ', X_val.shape)

np.save(path + "models/" + "compressed_X_val", X_val)
np.save(path + "models/" + "compressed_y_val", y_val)   # compressed_y_val : target = -np.log(np.abs(orig - pred))


# ---------------------------------- load training data --------------------------------------- 
print("Now creating training data")

num_training = 9*9920     #5952
X_train = np.zeros((num_training,64,1,1))
y_train = np.zeros((num_training,1,2,1))

noises = [2, 3, 4, 5, 10, 20, 30, 50, 100, 200]

Xis = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
Xis.remove(10)			# remove 10, 20, 30 and 40 value (This value will be used to create validation and test set)
Xis.remove(20)
Xis.remove(30)	
Xis.remove(40)

count = 0

for sigma in sigmas:
	for alpha in alphas:
		for Xi in Xis:
			for width in widths:
				for s in range(2):
					for noise in noises:
						space = math.floor(width*2**s)
						shift = math.floor(-25 + (width + space/2 + Xi + alpha*10 + sigma*10)%16) 
						
						noisy_file = path + 'images/' + 'noisy_images2/nim_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.tiff'
						linescan_file = path + 'images/' + 'linescans/linescan_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '.txt'
						
						linescan = []
						with open(linescan_file,'r') as f:
							for i,line in enumerate(f):
								if i < 3000:
									a, b = line.split(',')
									linescan.append(float(b))
								else:
									break
						linescan = linescan[:2048]
						leftline = np.array(linescan[:1024])
						rightline = linescan[1024:]
						rightline.reverse()
						rightline = np.array(rightline)
						leftline = leftline + shift
						rightline = rightline + shift
						
						# removed rounding
						lline_std = leftline.std()/2
						rline_std = rightline.std()/2
						
						imnoisy = np.array(Image.open(noisy_file))/256
						imnoisy = np.reshape(imnoisy, (1,1024,64,1))
						
						
						orig = np.array([[ [lline_std], [rline_std] ]])
						pred = EdgeNet.predict(imnoisy)
						
						target = -np.log(np.abs(orig - pred))
						
						X_train[count] = model.predict(imnoisy)
						y_train[count] = target
						count += 1					
print('Train_count: ',count)

X_train = np.reshape(X_train,(num_training,64))
y_train = np.reshape(y_train,(num_training,2))

print('Train data shape: ', X_train.shape)

np.save(path + "models/" + "compressed_X_train", X_train)
np.save(path + "models/" + "compressed_y_train", y_train)   # compressed_y_train : target = -np.log(np.abs(orig - pred))


print("Now loading data")

X_train = np.load(path + "models/" + "compressed_X_train.npy")
y_train = np.load(path + "models/" + "compressed_y_train.npy")

X_val = np.load(path + "models/" + "compressed_X_val.npy")
y_val = np.load(path + "models/" + "compressed_y_val.npy")



print("Now training")

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(64,)))
model.add(Dense(64, activation='relu', kernel_regularizer=L2(0.01))) 
model.add(Dense(64, activation='relu', kernel_regularizer=L2(0.01)))
model.add(Dense(2, activation='relu'))

model.summary()

adam = keras.optimizers.adam(lr=1e-3)
model.compile(loss='mse', optimizer=adam)


callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=9, min_lr=0.0001),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=18,
        ),
    # keras.callbacks.ModelCheckpoint(
        # path + 'models/' + 'Weights/compressedweights.{epoch:02d}-{val_loss:.2f}.hdf5',
        # monitor='val_loss',
        # save_best_only=True,
        # verbose=1),
]
	
history = model.fit(X_train, y_train,
             batch_size=16,
             epochs=24,
             validation_data=(X_val, y_val),
             shuffle=True,
			 callbacks=callbacks)
			  
model.save(path + 'models/' +'CPNormalizer.h5')

del model  # deletes the existing model


print("Execttion Time= ", time.time() - start)

