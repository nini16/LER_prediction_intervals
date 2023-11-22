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

model_left = load_model(path_asmc + 'models/' + 'NCPBLSTMNet_ledge.h5', compile=False)
model_left.compile('adam', loss='mae')
model_right = load_model(path_asmc + 'models/' + 'NCPBLSTMNet_redge.h5', compile=False)
model_right.compile('adam', loss='mae')



df = pd.DataFrame(columns = ['norm_log_left_error', 'norm_log_right_error'])

df_test = pd.DataFrame(columns = ['norm_log_left_error', 'norm_log_right_error'])


count = 0
test_count = 0
cal_count = 0

random_indexes = np.load(path + 'dataset/' + "random_indexes.npy")

input2 = np.zeros((1,2))

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
                        
                        left = model_left.predict(diff)[0]
                        right = model_right.predict(diff)[0]
                        
                        if count in random_indexes:
                            df_test.loc[test_count] = [left[0], right[0]]
                            test_count += 1
                        else:
                            df.loc[cal_count] = [left[0], right[0]]
                            cal_count += 1
print("Test count: ", test_count)

print(df_test.describe())

df_test.to_pickle(path_asmc + 'models/' + "Test_df_NORMCP_approach2")

print("Calibration count: ", cal_count)

print(df.describe())

df.to_pickle(path_asmc + 'models/' + "Calibration_df_NORMCP_approach2")



# Generate reults
df = pd.read_pickle(path + "models/Calibration_df_run5")
df_test = pd.read_pickle(path + "models/Test_df_run5")

df_norm2 = pd.read_pickle(path_asmc + 'models/' + "Calibration_df_NORMCP_approach2")
df_test_norm2 = pd.read_pickle(path_asmc + 'models/' + "Test_df_NORMCP_approach2")

df['norm_log_left_error2'] = df_norm2['norm_log_left_error']
df['norm_log_right_error2'] = df_norm2['norm_log_right_error']
df_test['norm_log_left_error2'] = df_test_norm2['norm_log_left_error']
df_test['norm_log_right_error2'] = df_test_norm2['norm_log_right_error']

miscoverage = 0.05
coverage_label = '95'

print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ {}% LEFT Test reults $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$".format(coverage_label))

cp_left_conf = df['lpred_nonconformity']/np.exp(-df['norm_log_left_error2'])
cp_left_conf = cp_left_conf.sort_values(ascending=False)
significance_index = math.floor(miscoverage*(len(cp_left_conf) + 1))

check = 0
interval_len = []
ratio = []

for i in range(len(df_test)):
    original = df_test.iloc[i]['i_leftline_sigma']
    pred = df_test.iloc[i]['ledge_pred_sigma']
    delta = cp_left_conf.iloc[significance_index]*np.exp(-df_test['norm_log_left_error2']).iloc[i]
    left = (pred-delta)
    right = (pred+delta)
    # print("original: {}, interval: [{} - {}]".format(original, left, right))
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


print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ {}% LEFT Test reults $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$".format(coverage_label))

cp_right_conf = df['rpred_nonconformity']/np.exp(-df['norm_log_right_error2'])
cp_right_conf = cp_right_conf.sort_values(ascending=False)
significance_index = math.floor(miscoverage*(len(cp_right_conf) + 1))

check = 0
interval_len = []
ratio = []

for i in range(len(df_test)):
    original = df_test.iloc[i]['i_rightline_sigma']
    pred = df_test.iloc[i]['redge_pred_sigma']
    delta = cp_right_conf.iloc[significance_index]*np.exp(-df_test['norm_log_right_error2']).iloc[i]
    left = (pred-delta)
    right = (pred+delta)
    # print("original: {}, interval: [{} - {}]".format(original, left, right))
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