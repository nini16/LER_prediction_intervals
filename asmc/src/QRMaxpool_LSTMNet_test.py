import keras.backend as K
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Lambda

import tensorflow_addons as tfa

from PIL import Image
import math
import pandas as pd
import numpy as np

# testing
sigmas = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]	
Xis    = [10, 20, 30, 40]
widths = [20, 30]
noises = [2, 3, 4, 5, 10, 20, 30, 50, 100, 200]



testsize = len(sigmas)*len(Xis)*len(alphas)*len(widths)*len(noises)*2
print('Testsize: ', testsize)

path = '/scratch/user/nini16/SEM/'
path_asmc = '/scratch/user/nini16/ASMC_SEM/'

sem_model = load_model(path + 'models/' + 'SEMNet_run2_epoch_4.h5')
edgenet_model = load_model(path + 'models/' + 'EDGEnet2_int_L1_epoch_4.h5', compile=False)
edgenet_model.add(Lambda(lambda x: K.std(x*64, axis=1)/2)) # removed rounding
ncplstmnet_left = load_model(path_asmc + 'models/' +'NCPLSTMNet_ledge.h5')
ncplstmnet_right = load_model(path_asmc + 'models/' +'NCPLSTMNet_redge.h5')
ncpmaxpoolnet = load_model(path + 'models/' + 'NCPMaxpool.h5')

model_left = load_model(path_asmc + 'models/' + 'QRMaxpool_LSTMNet_ledge_90.h5', compile=False)
model_left.compile('adam', loss=[tfa.losses.PinballLoss(tau=0.05), tfa.losses.PinballLoss(tau=0.95)])
model_right = load_model(path_asmc + 'models/' + 'QRMaxpool_LSTMNet_redge_90.h5', compile=False)
model_right.compile('adam', loss=[tfa.losses.PinballLoss(tau=0.05), tfa.losses.PinballLoss(tau=0.95)])




df = pd.DataFrame(columns = ['QR_noise_lower_left', 'QR_noise_upper_left', 'QR_noise_lower_right', 'QR_noise_upper_right'])

df_test = pd.DataFrame(columns = ['QR_noise_lower_left', 'QR_noise_upper_left', 'QR_noise_lower_right', 'QR_noise_upper_right'])
                             

count = 0
test_count = 0
cal_count = 0

random_indexes = np.load(path + 'dataset/' + "random_indexes.npy")

input2 = np.zeros((1,6))

for sigma in sigmas:
    for alpha in alphas:
        for Xi in Xis:
            for width in widths:
                for s in range(2):
                    count += 1
                    for noise in noises:
                        space = math.floor(width*2**s)
                        shift = math.floor(-25 + (width + space/2 + Xi + alpha*10 + sigma*10)%16) 
                        
                        # original_file = path + 'images/' + 'original_images2/oim_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '.tiff'
                        noisy_file = path + 'images/' + 'noisy_images2/nim_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.tiff'
                        
                        imnoisy = np.array(Image.open(noisy_file))
                        imnoisy = imnoisy/256
                        imnoisy = np.reshape(imnoisy, (1,1024,64,1))

                        denoised = sem_model.predict(imnoisy)
                        denoised = np.reshape(denoised, (1,1024,64,1))
                        diff = imnoisy-denoised
                        diff = np.abs(diff)
                        
                        left = ncplstmnet_left.predict(diff)[0]
                        right = ncplstmnet_right.predict(diff)[0]
                        
                        natlog5 = ncpmaxpoolnet.predict(diff)[0]
                        
                        ler_pred = edgenet_model.predict(imnoisy)[0]
                        
                        input2[0,0] = ler_pred[0][0]
                        input2[0,3] = ler_pred[1][0]
                        
                        input2[0,1] = left[0]
                        input2[0,4] = right[0]
                        
                        input2[0,2] = natlog5[0]
                        input2[0,5] = natlog5[1]
                        
                        left = np.array( model_left.predict(input2[:,:3]) ).reshape((2,))
                        right = np.array( model_right.predict(input2[:,3:]) ).reshape((2,))
                        
                        if count in random_indexes:
                            df_test.loc[test_count] = [left[0], left[1], right[0], right[1]]
                            test_count += 1
                        else:
                            df.loc[cal_count] = [left[0], left[1], right[0], right[1]]
                            cal_count += 1
print("Test count: ", test_count)

print(df_test.describe())

df_test.to_pickle(path_asmc + 'models/' + "Test_df_QR_approach3_90")

print("Calibration count: ", cal_count)

print(df.describe())

df.to_pickle(path_asmc + 'models/' + "Calibration_df_QR_approach3_90")



# Generate reults
df = pd.read_pickle(path + "models/Calibration_df_run5")
df_test = pd.read_pickle(path + "models/Test_df_run5")

df1 = pd.read_pickle(path_asmc + 'models/' + "Calibration_df_QR_approach3_90")
df_test1 = pd.read_pickle(path_asmc + 'models/' + "Test_df_QR_approach3_90")

df['QR_noise_lower_left'] = df1['QR_noise_lower_left']
df['QR_noise_upper_left'] = df1['QR_noise_upper_left']
df_test['QR_noise_lower_left'] = df_test1['QR_noise_lower_left']
df_test['QR_noise_upper_left'] = df_test1['QR_noise_upper_left']

df['QR_noise_lower_right'] = df1['QR_noise_lower_right']
df['QR_noise_upper_right'] = df1['QR_noise_upper_right']
df_test['QR_noise_lower_right'] = df_test1['QR_noise_lower_right']
df_test['QR_noise_upper_right'] = df_test1['QR_noise_upper_right']


print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ LEFT Test reults $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
check = 0
interval_len = []
ratio = []

for i in range(len(df_test)):
  original = df_test.iloc[i]['i_leftline_sigma']
  left = df_test.iloc[i]['QR_noise_lower_left']
  right = df_test.iloc[i]['QR_noise_upper_left']
  interval_len.append(right-left)
  ratio.append((right-left)/original)

  if (original <= right) and (original >= left):
    check += 1
print('coverage: ' + str(check/len(df_test)))
print('ave interval: ' + str(np.mean(interval_len)))
print('min interval: ' + str(min(interval_len)))
print('max interval: ' + str(max(interval_len)))
print('min ratio: ' + str(min(ratio)))
print('max ratio: ' + str(max(ratio)))
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")


print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ RIGHT Test reults $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
check = 0
interval_len = []
ratio = []

for i in range(len(df_test)):
  original = df_test.iloc[i]['i_rightline_sigma']
  left = df_test.iloc[i]['QR_noise_lower_right']
  right = df_test.iloc[i]['QR_noise_upper_right']
  interval_len.append(right-left)
  ratio.append((right-left)/original)

  if (original <= right) and (original >= left):
    check += 1
print('coverage: ' + str(check/len(df_test)))
print('ave interval: ' + str(np.mean(interval_len)))
print('min interval: ' + str(min(interval_len)))
print('max interval: ' + str(max(interval_len)))
print('min ratio: ' + str(min(ratio)))
print('max ratio: ' + str(max(ratio)))
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")