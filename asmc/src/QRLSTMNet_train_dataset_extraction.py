import keras
import keras.backend as K
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Lambda

import tensorflow_addons as tfa
import tensorflow as tf

import os
import numpy as np
from PIL import Image
import timeit,time
import math
import pandas as pd


############################# description ##########################################
# In this dataset we have 2 features: (edgenet_ler, NCPLSTMNet estimate).
##################################################################################

path = '/scratch/user/nini16/SEM/'
path_asmc = '/scratch/user/nini16/ASMC_SEM/'

X_train_path = path_asmc + "dataset/" + "X_train_QRLSTMNet.npy"
y_train_path = path_asmc + "dataset/" + "y_train_QR"

extract_data_flag = os.path.exists(X_train_path)

# models
sem_model = load_model(path + 'models/' + 'SEMNet_run2_epoch_4.h5')
edgenet_model = load_model(path + 'models/' + 'EDGEnet2_int_L1_epoch_4.h5', compile=False)
edgenet_model.add(Lambda(lambda x: K.std(x*64, axis=1)/2)) # removed rounding
ncplstmnet_left = load_model(path_asmc + 'models/' +'NCPLSTMNet_ledge.h5')
ncplstmnet_right = load_model(path_asmc + 'models/' +'NCPLSTMNet_redge.h5')

# Extraction
num_training = 9920*9
X_train      = np.zeros((num_training, 4))
y_train      = np.zeros((num_training, 2))

sigmas = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
Xis = [6,7,8,9,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29,31,32,33,34,35,36,37,38,39]
widths = [20, 30]
noises = [2, 3, 4, 5, 10, 20, 30, 50, 100, 200]

print('###################### Begin Extraction ###############################')

count = 0
for sigma in sigmas:
    for alpha in alphas:
        print("working on alpha: {}".format(alpha))
        for Xi in Xis:
            for width in widths:
                for s in range(2):
                    for noise in noises:
                        space = math.floor(width*2**s)
                        shift = math.floor(-25 + (width + space/2 + Xi + alpha*10 + sigma*10)%16)
                        
                        noisy_file = path + 'images/' + 'noisy_images2/nim_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.tiff'
                        linescan_file = path + 'images/' + 'linescans/linescan_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '.txt'
                        
                        imnoisy = np.array(Image.open(noisy_file))
                        imnoisy = imnoisy/256
                        imnoisy = np.reshape(imnoisy, (1,1024,64,1))
                        
                        denoised = sem_model.predict(imnoisy)
                        denoised = np.reshape(denoised, (1,1024,64,1))
                        diff = imnoisy-denoised
                        diff = np.abs(diff)
                        
                        left = ncplstmnet_left.predict(diff)[0]
                        right = ncplstmnet_right.predict(diff)[0]
                        
                        ler_pred = edgenet_model.predict(imnoisy)[0]
                        X_train[count,0] = ler_pred[0][0]
                        X_train[count,2] = ler_pred[1][0]
                        
                        X_train[count,1] = left[0]
                        X_train[count,3] = right[0]
                        
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
                        
                        y_train[count, 0] = leftline.round().std()/2
                        y_train[count, 1] = rightline.round().std()/2
                        count += 1
    print("finished sigma: {}".format(sigma))
    print("count: {}".format(count))

np.save(X_train_path, X_train)
np.save(y_train_path, y_train)