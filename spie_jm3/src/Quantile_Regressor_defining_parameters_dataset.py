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

model = load_model(path + 'models/' + 'EDGEnet2_int_L1_epoch_4.h5')
model.add(Lambda(lambda x: K.std(x*64, axis=1)/2))


# DATASET GENERATION FOR IDEAL/DEFINING PARAMETERS EXPERIMENT

# ---------------------------------- generating validation data ---------------------------------------

num_validation = 2880
num_test = 8640

# Format of ideal/defining parameters:
# [sigma, alpha, Xi, width, space, noise, LER_L, LER_R]

ideal_X_val = np.zeros((num_validation,8))
ideal_y_val = np.zeros((num_validation,2))


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
						
						# original_file = path + 'images/' + 'original_images2/oim_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '.tiff'
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
						
						imnoisy = np.array(Image.open(noisy_file))
						imnoisy = imnoisy/256
						imnoisy = np.reshape(imnoisy, (1,1024,64,1))
						ler_pred = model.predict(imnoisy)[0]
						
						leftline = leftline + shift
						rightline = rightline + shift
						
						# removed rounding
						lline_std = leftline.std()/2
						rline_std = rightline.std()/2
						
						ideal_X_val[count, 0] = sigma
						ideal_X_val[count, 1] = alpha
						ideal_X_val[count, 2] = Xi
						ideal_X_val[count, 3] = width
						ideal_X_val[count, 4] = space
						ideal_X_val[count, 5] = noise
						ideal_X_val[count, 6] = ler_pred[0][0]
						ideal_X_val[count, 7] = ler_pred[1][0]
						
						
						ideal_y_val[count,0] = lline_std
						ideal_y_val[count,1] = rline_std
						count += 1					
print('Validation_count: ',count)

print('Validation data shape: ', ideal_y_val.shape)

np.save(path + "models/" + "ideal_X_val", ideal_X_val)
np.save(path + "models/" + "ideal_y_val", ideal_y_val)



# ---------------------------------- generating training data ---------------------------------------
print("Now creating training data")

num_training = 9*9920     

noises = [2, 3, 4, 5, 10, 20, 30, 50, 100, 200]

Xis = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
Xis.remove(10)			# remove 10, 20, 30 and 40 value (This value will be used to create validation and test set)
Xis.remove(20)
Xis.remove(30)	
Xis.remove(40)

ideal_X_train = np.zeros((num_training,8)) # [sigma, alpha, Xi, width, space, noise, LER_L, LER_R]
ideal_y_train = np.zeros((num_training,2))

count = 0


for sigma in sigmas:
	for alpha in alphas:
		for Xi in Xis:
			for width in widths:
				for s in range(2):
					for noise in noises:
						space = math.floor(width*2**s)
						shift = math.floor(-25 + (width + space/2 + Xi + alpha*10 + sigma*10)%16) 
						
						# original_file = path + 'images/' + 'original_images2/oim_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '.tiff'
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
						
						# im = np.array(Image.open(original_file))
						imnoisy = np.array(Image.open(noisy_file))
						imnoisy = imnoisy/256
						imnoisy = np.reshape(imnoisy, (1,1024,64,1))
						ler_pred = model.predict(imnoisy)[0]
						
						leftline = leftline + shift
						rightline = rightline + shift
						
						# removed rounding
						lline_std = leftline.std()/2
						rline_std = rightline.std()/2
						
						ideal_X_train[count, 0] = sigma
						ideal_X_train[count, 1] = alpha
						ideal_X_train[count, 2] = Xi
						ideal_X_train[count, 3] = width
						ideal_X_train[count, 4] = space
						ideal_X_train[count, 5] = noise
						ideal_X_train[count, 6] = ler_pred[0][0]
						ideal_X_train[count, 7] = ler_pred[1][0]
						
						
						ideal_y_train[count,0] = lline_std
						ideal_y_train[count,1] = rline_std
						count += 1					
print('Validation_count: ',count)

print('Training data shape: ', ideal_y_train.shape)

np.save(path + "models/" + "ideal_X_train", ideal_X_train)
np.save(path + "models/" + "ideal_y_train", ideal_y_train)