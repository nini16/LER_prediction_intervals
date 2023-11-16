from keras.layers import Reshape, Lambda
from keras import backend as K

import keras
import keras.backend as K
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D
from keras.constraints import max_norm

import tensorflow_addons as tfa
import numpy as np
import matplotlib as plt
from keras.models import Sequential, load_model, Model
from PIL import Image
import timeit,time
import math
import pandas as pd
import matplotlib.pyplot as plt

from skimage import measure
from scipy import ndimage as ndi
from skimage import feature

path = '/scratch/user/nini16/SEM/'

# AutoEncoder model
enc = load_model(path + 'models/' + 'AutoEncoder3_norm_run_1_epoch_4.h5')
enc = Model(inputs=enc.input, outputs=enc.get_layer('conv2d_7').output)

model = load_model(path + 'models/' + 'CPNormalizer.h5')


# testing
sigmas = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]	
Xis    = [10, 20, 30, 40]
widths = [20, 30]
noises = [2, 3, 4, 5, 10, 20, 30, 50, 100, 200]


testsize = len(sigmas)*len(Xis)*len(alphas)*len(widths)*len(noises)*2
print('Testsize: ', testsize)

df = pd.DataFrame(columns = ['norm_log_left_error', 'norm_log_right_error'])

df_test = pd.DataFrame(columns = ['norm_log_left_error', 'norm_log_right_error'])
                             

count = 0
test_count = 0
cal_count = 0

random_indexes = np.load(path + 'models/' + "random_indexes.npy")

for sigma in sigmas:
	for alpha in alphas:
		for Xi in Xis:
			for width in widths:
				for s in range(2):
					count += 1
					for noise in noises:
						space = math.floor(width*2**s)
						shift = math.floor(-25 + (width + space/2 + Xi + alpha*10 + sigma*10)%16) 

						noisy_file = path + 'images/' + 'noisy_images2/nim_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.tiff'
						
						imnoisy = np.array(Image.open(noisy_file))/256
						imnoisy = np.reshape(imnoisy, (1,1024,64,1))
						
						encoded = enc.predict(imnoisy)
						encoded = np.reshape(encoded, (1, 64))
						
						log_err = model.predict(encoded)[0]

						if count in random_indexes:
							df_test.loc[test_count] = [log_err[0], log_err[1]]
							test_count+=1
						else:
							df.loc[cal_count] = [log_err[0], log_err[1]]
							cal_count += 1
						    
                        
print("Test count: ", test_count)

print(df_test.describe())

df_test.to_pickle(path + 'models/' + "Test_df_normcp")

print("Calibration count: ", cal_count)

print(df.describe())

df.to_pickle(path + 'models/' + "Calibration_df_normcp")		  


