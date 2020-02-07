# Restrict to one GPU in case there are several GPUs available
import imp
try:
	imp.find_module('setGPU')
	import setGPU
except ImportError:
	found = False

import time
import numpy as np
import pandas as pd
import root_numpy
import root_pandas

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from Plots_Losses import plot_losses

from Disco_tf import distance_corr

# mass_point_list = ['15_10', '19_01', '19_08', '19_10', '22_01', '22_08']
mass_point_list = ['15_10']

# Choose the lambda hyperparameter
lam = 2.

# Train a neural network separately for each mass point
for mass_point in mass_point_list:
	print 'Training mass point: ' + str(mass_point)

	data_dir = '/work/kimmokal/susyDNN/preprocessedData/'
	train_path = data_dir+'train_set_all_bkg_nJets_'+str(mass_point)+'.root'
	test_path = data_dir+'test_set_all_bkg_nJets_'+str(mass_point)+'.root'

	# Read the ROOT files for background and signal samples and put them into dataframes
	train = root_pandas.read_root(train_path, 'tree')
	test = root_pandas.read_root(test_path, 'tree')

	# Drop the sample names and weights at this point
	train = train.drop(columns=['sampleName'])
	test = test.drop(columns=['sampleName'])

	# Separate input features and the target
	train_y = train['target']
	test_y = test['target']

	train_x = train.drop(columns=['target'])
	test_x = test.drop(columns=['target'])

	decorr_var_col = train_x.columns.get_loc('nJets30Clean')

	# Split train/val
	train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

	batchSize = 2048

	train_x_tf = tf.data.Dataset.from_tensor_slices(train_x.to_numpy().astype(np.float32))
	train_y_tf =tf.data.Dataset.from_tensor_slices(train_y.to_numpy().astype(np.float32))
	train_data = tf.data.Dataset.zip((train_x_tf, train_y_tf))
	train_data_batched = train_data.batch(batchSize)

	val_x_tf = tf.data.Dataset.from_tensor_slices(val_x.to_numpy().astype(np.float32))
	val_y_tf =tf.data.Dataset.from_tensor_slices(val_y.to_numpy().astype(np.float32))
	val_data = tf.data.Dataset.zip((val_x_tf, val_y_tf))
	val_data_batched = val_data.batch(batchSize)

	### Build the neural network ###
	from tensorflow.keras import layers
	from sklearn.utils import class_weight

	# Define the architecture
	classifier_inputs = layers.Input(shape = (train_x.shape[1], ))
	classifier_hidden = layers.Dense(100, kernel_initializer='normal', activation='relu')(classifier_inputs)
	classifier_hidden = layers.Dropout(0.2)(classifier_hidden)
	classifier_hidden = layers.Dense(100, kernel_initializer='normal', activation='relu')(classifier_hidden)
	classifier_hidden = layers.Dropout(0.2)(classifier_hidden)
	classifier_hidden = layers.Dense(50, kernel_initializer='normal', activation='relu')(classifier_hidden)
	classifier_out = layers.Dense(1, kernel_initializer='normal', name="out_classifier", activation='sigmoid')(classifier_hidden)
	classifier_model = tf.keras.Model(inputs=[classifier_inputs], outputs=[classifier_out])

	opt_model = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)

	# Define the custom loss function
	import tensorflow.keras.backend as kb
	def make_disco_loss(decorr_var):
		def disco_loss(y_true, y_pred):
			weights = tf.ones(decorr_var.shape, dtype=tf.dtypes.float32)
			return kb.binary_crossentropy(y_true, y_pred) + lam*distance_corr(decorr_var, y_pred, normedweight=weights, power=1)
		return disco_loss

	from Plots_Losses_DisCo import plot_losses
	losses = {"L_t": [], "L_v": []} # L_t = training loss, L_v = validation loss

	numberOfEpochs = 200
	for epoch in range(numberOfEpochs):
		epoch_losses = []
		val_epoch_losses = []
		epoch_start_time = time.time()
		for (x_batch, y_batch) in train_data_batched:
			classifier_model.compile(loss=make_disco_loss(x_batch.numpy()[:,decorr_var_col]), optimizer=opt_model, metrics=['accuracy'])
			history = classifier_model.fit(x_batch, y_batch,  epochs=1, batch_size=batchSize, verbose=0)
			epoch_losses.append(history.history['loss'])
			tf.keras.backend.clear_session()
		print 'Epoch mean loss: ', round(np.mean(epoch_losses), 5)
		for (x_batch, y_batch) in val_data_batched:
			classifier_model.compile(loss=make_disco_loss(x_batch.numpy()[:,decorr_var_col]), optimizer=opt_model, metrics=['accuracy'])
			val_loss, val_acc = classifier_model.evaluate(x_batch, y_batch, batch_size=batchSize)
			val_epoch_losses.append(val_loss)
			tf.keras.backend.clear_session()
		print 'Epoch validation mean loss: ',  round(np.mean(val_epoch_losses), 5)
		losses["L_t"].append(np.mean(epoch_losses))
		losses["L_v"].append(np.mean(val_epoch_losses))
		plot_losses(epoch, losses, lam, numberOfEpochs, 'Losses_DisCo_all_bkg_lambda_'+str(lam))
		tf.keras.backend.clear_session()
		epoch_end_time = time.time()
		print 'Epoch time elapsed: ', np.round((epoch_end_time - epoch_start_time), 3)
		print "End of epoch: ", epoch+1
		print '-------------'

	print ' - Test set ROC AUC: ', round(roc_auc_score(test_y,classifier_model.predict(test_x)), 4)

	# Save the model
	save_path = '/work/kimmokal/susyDNN/models/'
	save_name = 'susyDNN_DisCo_all_bkg_nJets_lambda'+str(lam)+'_'+str(mass_point)
	classifier_model.save(save_path+save_name+'.h5')
