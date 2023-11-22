import tensorflow as tf

import numpy as np
import math
import timeit

import keras
from keras.models import Sequential,load_model,Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Input, Concatenate
from keras.constraints import max_norm
from PIL import Image
import keras.backend as K
from multi_gpu import make_parallel
import time
from random import shuffle

start = time.time()

path = '/scratch/user/nini16/SEM/'
# SEMNET
sem_model = load_model(path + 'models/' + 'SEMNet_run2_epoch_4.h5')
sem_model.summary()

#getting the data

num_validation = 2880
num_test = 8640

X_val = np.zeros((num_validation,1024,64,1))

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
						imnoisy = imnoisy/256
						imnoisy = np.reshape(imnoisy, (1,1024,64,1))
						
						denoised = sem_model.predict(imnoisy)
						
						denoised = np.reshape(denoised, (1,1024,64,1))
						
						diff = imnoisy-denoised
						
						diff = np.abs(diff)
						
						# diff = np.reshape(diff, (1024, 64))
						
						X_val[count] = diff
						count += 1
print('Validation_count: ',count)


np.save(path + "models/" + "NoiseCompression_Xval", X_val)

print('Validation data shape: ', X_val.shape)
	

batch_size = 9



# ----------------------------------load training data and train on it --------------------------------------- 

num_training = 9920*9
X_train = np.zeros((num_training,1024,64,1)) 

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
						
						imnoisy = np.array(Image.open(noisy_file))/256
					
						imnoisy = np.reshape(imnoisy, (1,1024,64,1))
						denoised = sem_model.predict(imnoisy)
						denoised = np.reshape(denoised, (1,1024,64,1))
						
						diff = imnoisy-denoised
						
						diff = np.abs(diff)
						
						# diff = np.reshape(diff, (1024, 64))
						
						X_train[count] = diff
						count += 1
print('Train_count: ',count)


np.save(path + "models/" + "NoiseCompression_Xtrain", X_train)

print('Train data shape: ', X_train.shape)



print("Execttion Time= ", time.time() - start)


