import tensorflow as tf
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
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D
from keras.constraints import max_norm
from PIL import Image
import keras.backend as K
from multi_gpu import make_parallel
import time
from random import shuffle

start = time.time()


# ----------------------------------load validation data for training ---------------------------------------

num_validation = 2880
num_test = 8640

X_val = np.zeros((num_validation,1024,64))
y_val = np.zeros((num_validation,1024,64))


path = '/scratch/user/nini16/SEM/'
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
						
						
						imnoisy = np.array(Image.open(noisy_file))

						X_val[count] = imnoisy
						y_val[count] = imnoisy
						count += 1
print('Validation_count: ',count)

X_val = X_val/256
y_val = y_val/256
X_val = np.reshape(X_val,(num_validation,1024,64,1))
y_val = np.reshape(y_val,(num_validation,1024,64,1))	

print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)

	

batch_size = 8
epochs = 4

# AutoEncoder model
model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same',input_shape= (1024,64,1), activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same', strides=(1, 2),activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same', strides=(1, 2), activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(512, (3, 3), padding='same', strides=(1, 2), activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(1, (3, 3), padding='same'))

#############  Middle Compressed representation ##################
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(1, (3, 3), strides=(2, 2), padding='same'))

model.add(MaxPooling2D((2,2)))
model.add(Conv2D(1, (3, 3), strides=(2, 1), padding='same'))
##################################################################

model.add(UpSampling2D(size=(4, 4)))

model.add(Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(UpSampling2D(size=(2, 4)))

model.add(Conv2D(256, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(UpSampling2D(size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(UpSampling2D(size=(1, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(1, (3, 3), strides=(1, 1), padding='same'))



model.summary()

G = 1
if G > 1:
   model = make_parallel(model,G)


adam = keras.optimizers.adam(lr=1e-3)

model.compile(loss = 'mean_squared_error',
              optimizer=adam)


# ----------------------------------load training data and train on it --------------------------------------- 

num_training = 9920
X_train = np.zeros((num_training,1024,64,1))
y_train = np.zeros((num_training,1024,64,1))

noises = [2, 3, 4, 5, 10, 20, 30, 50, 100, 200]

Xis = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
Xis.remove(10)			# remove 10, 20, 30 and 40 value (This value will be used to create validation and test set)
Xis.remove(20)
Xis.remove(30)	
Xis.remove(40)


for epoch in range(1, epochs+1):
	shuffle(alphas)
	for alpha in alphas:
		count = 0
		for sigma in sigmas:
			for noise in noises:
				for Xi in Xis:
					for width in widths:
						for s in range(2):
							space = math.floor(width*2**s)
							shift = math.floor(-25 + (width + space/2 + Xi + alpha*10 + sigma*10)%16) 
							
							noisy_file = path + 'images/' + 'noisy_images2/nim_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.tiff'
							
							imnoisy = np.array(Image.open(noisy_file))
							
							imnoisy = imnoisy/256
							imnoisy = np.reshape(imnoisy,(1024,64,1))
							X_train[count] = imnoisy
							y_train[count] = imnoisy
							count += 1
		print("alpha set, Training count :",alpha,',',count)
		history = model.fit(X_train, y_train, batch_size=batch_size, epochs=1, shuffle=True)
	print('Running validation now for epoch ' + str(epoch))
	val_score = model.evaluate(X_val,y_val)
	print('Validation score:',val_score)
	model.save(path + 'models/' +'AutoEncoder3_norm_run_1_epoch_' + str(epoch) + '.h5')
			  

del model  # deletes the existing model


print("Execttion Time= ", time.time() - start)

