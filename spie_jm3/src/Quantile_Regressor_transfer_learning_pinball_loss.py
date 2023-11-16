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


# Custom pinball loss functions. Since Tensorflow addons are not always backward compatible
# with Keras/Tensorflow libraries it might make sense to define your own
def pinball_loss(y_true, y_pred, tau):
	err = y_true - y_pred
	return K.mean(K.max(tau*err, (tau - 1)*err), axis=-1)


class MyPinballLoss(Loss): #inherit parent class
	#class attribute
	quant = 0.9

	#initialize instance attributes
	def __init__(self, quant):
		super().__init__()
		self.quat = quant

	#compute loss
	def call(self, y_true, y_pred):
		tau = self.quant
		y_pred = tf.convert_to_tensor(y_pred)
		y_true = tf.cast(y_true, y_pred.dtype)

		# Broadcast the pinball slope along the batch dimension
		tau = tf.expand_dims(tf.cast(tau, y_pred.dtype), 0)
		one = tf.cast(1, tau.dtype)
		delta_y = tf.subtract(y_true, y_pred)
		# pinball = tf.math.maximum(tau * delta_y, (tau - one) * delta_y)
		pinball = delta_y
		return tf.reduce_mean(pinball, axis=-1)

# def reset_weights(model):
# 	session = tf.compat.v1.Session()
# 	for layer in model.layers:
# 		if hasattr(layer,'init'):
# 			input_dim = layer.input_shape[1]
# 			new_weights = layer.init((input_dim, layer.output_dim),name='{}_W'.format(layer.name))
# 			layer.trainable_weights[0].set_value(new_weights.get_value())


start = time.time()


#getting the data

# QUANTILE REGRESSION

num_validation = 2880	#1728		# will be 2880 in full set
num_test = 8640

X_val = np.zeros((num_validation,1024,64))
y_val = np.zeros((num_validation,2))


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
						
						# im = np.array(Image.open(original_file))
						imnoisy = np.array(Image.open(noisy_file))
						leftline = leftline + shift
						rightline = rightline + shift
						
						# removed rounding
						lline_std = leftline.std()/2
						rline_std = rightline.std()/2
						
						X_val[count] = imnoisy
						y_val[count,0] = lline_std
						y_val[count,1] = rline_std
						count += 1					
print('Validation_count: ',count)

X_val = X_val/256

X_val = np.reshape(X_val,(num_validation,1024,64,1))
y_val = np.reshape(y_val,(num_validation,2,1))	

print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)

	

batch_size = 10
epochs = 4


model = load_model(path + 'models/' + 'QuantileReg_lower_90_epoch_4.h5', compile=False)


train_flag = False
for layer in model.layers:
    if layer.name == "conv2d_15": # adjust based on the number of layers to freeze for training
        train_flag = True
    layer.trainable = train_flag


model.summary()

G = 1
if G > 1:
   model = make_parallel(model,G)


adam = keras.optimizers.adam(lr=1e-3)

# train lower quantile model at 90% confidence
quantile = 0.05

# You can use the Pinball loss function in tensorflow addons package or define your custom loss function
# MyPinballLoss(quantile), # lambda y, y_hat: pinball_loss(y, y_hat, quantile)
model.compile(loss = tfa.losses.PinballLoss(tau=quantile),
              optimizer=adam)



# ----------------------------------load training data and train on it --------------------------------------- 

num_training = 9920     #5952
X_train = np.zeros((num_training,1024,64,1))
y_train = np.zeros((num_training,2,1))

noises = [2, 3, 4, 5, 10, 20, 30, 50, 100, 200]
#noises = [2, 3, 4, 5, 10, 20]

Xis = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
Xis.remove(10)			# remove 10, 20, 30 and 40 value (This value will be used to create validation and test set)
Xis.remove(20)
Xis.remove(30)	
Xis.remove(40)


for epoch in range(5, 5+epochs+1):
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
							
							# original_file = path + 'images/' + 'original_images2/oim_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '.tiff'
							noisy_file = path + 'images/' + 'noisy_images2/nim_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.tiff'
							linescan_file = path + 'images/' + 'linescans/linescan_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '.txt'
							
							
							imnoisy = np.array(Image.open(noisy_file))
							imnoisy = imnoisy/256
							imnoisy = np.reshape(imnoisy,(1024,64,1))
							linescan = []
							#print(linescan_file)
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

							X_train[count] = imnoisy
							y_train[count,0][0] = lline_std
							y_train[count,1][0] = rline_std
							count += 1
		print("alpha set, Training count :",alpha,',',count)
		history = model.fit(X_train, y_train, batch_size=batch_size, epochs=1, shuffle=True)
		model.save(path + 'models/' +'QuantileReg_lower_90_epoch_' + str(epoch) + '.h5') # lower quantile for 90% confidence
	print('Running validation now for epoch ' + str(epoch))
	val_score = model.evaluate(X_val,y_val)
	print('Validation score:',val_score)
	model.save(path + 'models/' +'QuantileReg_lower_90_epoch_' + str(epoch) + '.h5')


del model  # deletes the existing model

print("Execttion Time= ", time.time() - start)

