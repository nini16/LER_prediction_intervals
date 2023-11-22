import os
import numpy as np


############################# description ##########################################
# In this dataset we had 3 features: (edgenet_ler, NCPBLSTMNet estimate, NCPMaxpool estimate).
##################################################################################

path_asmc = '/scratch/user/nini16/ASMC_SEM/'

data3 = np.load('X_train_QRMaxpool_LSTMNet.npy')  # (edgenet_ler, NCPLSTMNet estimate, NCPMaxpool estimate)
data4 = np.load('X_train_QRLSTM_BLSTMNet.npy')  # (edgenet_ler, NCPLSTMNet estimate, NCPBLSTMNet estimate)

data_new = np.concatenate((data4[:,[0, 2]], data3[:,[2]], data4[:,[3, 5]], data3[:,[5]]), axis=1) # (edgenet_ler, NCPMaxpool estimate, NCPMaxpool estimate)
print(data_new.shape)

np.save('dataset/' + 'X_train_QRMaxpool_BLSTMNet.npy', data_new)