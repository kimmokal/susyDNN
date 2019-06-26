# Restrict to one GPU in case there are several GPUs available
import imp
try:
	imp.find_module('setGPU')
	import setGPU
except ImportError:
	found = False

import numpy as np
import pandas as pd
import root_numpy
import root_pandas

import tensorflow as tf
sess = tf.Session()
import keras.backend as K
K.set_session(sess)
import keras.callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

mass_point_list = ['15_10', '19_01', '19_08', '19_10', '22_01', '22_08']

# Train a neural network separately for each mass point
for mass_point in mass_point_list:
	print 'Training mass point: ' + str(mass_point)

	data_dir = '/work/kimmokal/susyDNN/preprocessedData/'
	train_path = data_dir+'train_set_'+str(mass_point)+'.root'
	test_path = data_dir+'test_set_'+str(mass_point)+'.root'

	# Read the ROOT files for background and signal samples and put them into dataframes
	train = root_pandas.read_root(train_path, 'tree')
	test = root_pandas.read_root(test_path, 'tree')

	# Drop the sample names at this point
	train = train.drop(columns=['sampleName'])
	test = test.drop(columns=['sampleName'])

	# Separate input features and the target
	train_y = train['target']
	test_y = test['target']

	train_x = train.drop(columns=['target'])
	test_x = test.drop(columns=['target'])

	### Build the neural network ###
	from keras.models import Sequential,Model
	from keras.layers import Input,Dense,Activation,Dropout
	from keras import optimizers
	from sklearn.utils import class_weight

	# Define the architecture
	model = Sequential()
	model.add(Dense(100, kernel_initializer='normal', activation='relu', input_dim=train_x.shape[1]))
	model.add(Dropout(0.2))
	model.add(Dense(100, kernel_initializer='normal', activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(50, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

	# Choose the optimizer and the loss function
	optimizer_ = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
	loss_ = 'binary_crossentropy'

	# Compile the model
	model.compile(optimizer=optimizer_, loss=loss_, metrics=['accuracy'])

	# Train the model with chosen hyperparameters
	classWeight = class_weight.compute_class_weight('balanced', np.unique(train_y), train_y[:])
	numberOfEpochs = 40
	batchSize = 256
	earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10)

	model.fit(train_x, train_y,
	        epochs=numberOfEpochs,
	        batch_size = batchSize,
	        class_weight=classWeight,
	        callbacks=[earlystop],
	        validation_split=0.1,
	        shuffle=True)

    # Save the model
	save_path = '/work/kimmokal/susyDNN/models/'
	save_name = 'susyDNN_model_'+str(mass_point)
	model.save(save_path+save_name+'.h5')

	# Print final test set scores
	test_loss, test_acc = model.evaluate(test_x, test_y, verbose=0)
	print ' - test set loss: ', round(test_loss, 4)
	print ' - test set accuracy: ', round(test_acc, 4)
	print ' - test set roc auc: ', round(roc_auc_score(test_y,model.predict(test_x)), 4)
