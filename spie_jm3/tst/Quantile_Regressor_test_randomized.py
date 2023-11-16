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

model = load_model(path + 'models/' + 'EDGEnet2_int_L1_epoch_4.h5')

# Load the model files for both lower and upper quantile level for a given confidence level
# Note that for certain versions of keras/tensorflow, custom objects like Tensorflow addons pinball loss function might
# not load correctly. In such cases set 'compile` to false when calling load_model.
model_upper = load_model(path + 'models/' + 'QuantileReg_upper_90_epoch_3.h5', custom_objects={'PinballLoss': tfa.losses.PinballLoss()})
model_lower = load_model(path + 'models/' + 'QuantileReg_lower_90_epoch_3.h5', custom_objects={'PinballLoss': tfa.losses.PinballLoss()})
# model_upper = load_model(path + 'models/' + 'QuantileReg_upper_95_smoothed_run2_epoch_2.h5', compile=False)
# model_lower = load_model(path + 'models/' + 'QuantileReg_lower_95_epoch_3.h5', compile=False)

adam = keras.optimizers.adam(lr=1e-3)

model_upper.compile(loss = tfa.losses.PinballLoss(tau=0.975), # MyPinballLoss(quantile), # lambda y, y_hat: pinball_loss(y, y_hat, quantile),
              optimizer=adam)
model_lower.compile(loss = tfa.losses.PinballLoss(tau=0.025), optimizer=adam)


# testing
sigmas = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]	
Xis    = [10, 20, 30, 40]
widths = [20, 30]
noises = [2, 3, 4, 5, 10, 20, 30, 50, 100, 200]


testsize = len(sigmas)*len(Xis)*len(alphas)*len(widths)*len(noises)*2
print('Testsize: ', testsize)

df = pd.DataFrame(columns = ['noise', 'sigma', 'alpha', 'Xi', 'width', 'space', 'MSE_noise', \
                             'PSNR_noise', 'Pred_time', 'i_leftline_sigma', 'ledge_pred_sigma',\
                             'i_rightline_sigma',  'redge_pred_sigma',\
                              'lpred_error', 'rpred_error', 'lpred_nonconformity', 'rpred_nonconformity',\
                             'lower_left_LER', 'upper_left_LER', 'lower_right_LER', 'upper_right_LER'])

df_test = pd.DataFrame(columns = ['noise', 'sigma', 'alpha', 'Xi', 'width', 'space', 'MSE_noise', \
                             'PSNR_noise', 'Pred_time', 'i_leftline_sigma', 'ledge_pred_sigma',\
                             'i_rightline_sigma',  'redge_pred_sigma',\
                              'lpred_error', 'rpred_error', 'lpred_nonconformity', 'rpred_nonconformity',\
                             'lower_left_LER', 'upper_left_LER', 'lower_right_LER', 'upper_right_LER'])
                             

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
						linescan_file = path + 'images/' + 'linescans/linescan_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '.txt'
						linescan = []

						with open(linescan_file,'r') as f:
							for i, line in enumerate(f):
								if i < 3000:
									a, b = line.split(',')
									linescan.append(float(b))

						linescan = linescan[:2048]

						leftline = np.array(linescan[:1024]) 
						rightline = linescan[1024:]
						rightline.reverse()
						rightline = np.array(rightline)

						leftline = leftline + shift           # add shift to linescan
						rightline = rightline + shift         # add shift to linescan

						imnoisy = np.array(Image.open(noisy_file))

						imnoisy = imnoisy/256
						imnoisy = imnoisy.reshape(1,1024,64,1)

						start = time.time()
						linepredict = model.predict(imnoisy)

						ler_lower = model_lower.predict(imnoisy)[0]
						ler_upper = model_upper.predict(imnoisy)[0]

						prediction_time = time.time() - start

						linepredict = linepredict.reshape(1024,2)

						imnoisy = imnoisy.reshape(1024,64)
						imnoisy = imnoisy.astype(float)

						mse_noisy = 0

						psnr_noisy = 0


						ledge_pred = (linepredict*64)[:,0].round()
						redge_pred = (linepredict*64)[:,1].round()


						lline_std = leftline.std()/2  # no need to round()
						rline_std = rightline.std()/2 # no need to round()
						lpred_std = (ledge_pred).std()/2
						rpred_std = (redge_pred).std()/2
						lerror = (lline_std - lpred_std)*100/lline_std
						rerror = (rline_std - rpred_std)*100/rline_std

						left_nonconformity = abs(lline_std - lpred_std)
						right_nonconformity = abs(rline_std - rpred_std)


						if count in random_indexes:
							df_test.loc[test_count] = [noise, sigma, alpha, Xi, width, space, mse_noisy, \
											 psnr_noisy, prediction_time, \
											 lline_std, lpred_std,\
											 rline_std, rpred_std,\
											 lerror, rerror, left_nonconformity, right_nonconformity,\
											 ler_lower[0][0], ler_upper[0][0], ler_lower[1][0], ler_upper[1][0]]
							test_count+=1
						else:
							df.loc[cal_count] = [noise, sigma, alpha, Xi, width, space, mse_noisy, \
										 psnr_noisy, prediction_time, \
										 lline_std, lpred_std,\
										 rline_std, rpred_std,\
										 lerror, rerror, left_nonconformity, right_nonconformity,\
										 ler_lower[0][0], ler_upper[0][0], ler_lower[1][0], ler_upper[1][0]]
							cal_count += 1
						    
# I like to save test results to pndas dataframe objects and store on file
# but there's plenty other ways to do this :)

print("Test count: ", test_count)

print(df_test.describe())

df_test.to_pickle(path + 'models/' + "Test_df_run5")
# df_test.to_pickle(path + 'models/' + "Test_df_run_95")

print("Calibration count: ", cal_count)

print(df.describe())

df.to_pickle(path + 'models/' + "Calibration_df_run5")
# df.to_pickle(path + 'models/' + "Calibration_df_run_95")
