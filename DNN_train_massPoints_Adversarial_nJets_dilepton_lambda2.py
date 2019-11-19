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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from Plots_Losses import plot_losses

# mass_point_list = ['15_10', '19_01', '19_08', '19_10', '22_01', '22_08']
mass_point_list = ['15_10']

# Choose the lambda hyperparameter
lam = 2.

# Train a neural network separately for each mass point
for mass_point in mass_point_list:
	print 'Training mass point: ' + str(mass_point)

	data_dir = '/work/kimmokal/susyDNN/preprocessedData/'
	train_path = data_dir+'train_set_dilepton_nJets_'+str(mass_point)+'.root'
	test_path = data_dir+'test_set_dilepton_nJets_'+str(mass_point)+'.root'

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


	### NJETS Bins ##############
	NJETS_BINS = np.array([0., 0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 1])  #Description: nJet>=4 && dPhi>0.5 | nJet bins=4,5,6,7,8,9,>=10 (7 bins in total)

	nJetsData = train["nJets30Clean"].values.flatten()
	nJetsData_test = test["nJets30Clean"].values.flatten() # Test set

	nJetsBins = np.digitize(nJetsData, NJETS_BINS)-1
	nJetsBins_test = np.digitize(nJetsData_test,  NJETS_BINS)-1
	## to_categorical ############
	nJetsBins_cat = keras.utils.to_categorical(nJetsBins)
	nJetsBins_cat_test = keras.utils.to_categorical(nJetsBins_test)
	###################################

	nJetsClasses = np.zeros((len(nJetsBins), len(NJETS_BINS)))
	for i in range(len(nJetsData)):
		nJetsClasses[i, nJetsBins[i]] = 1

	#### Convert from np array to pandas DataFrame ###
	column_to_be_added = np.arange(len(nJetsData))
	column_to_be_added_test = np.arange(len(nJetsData_test))

    #column_to_be_added.astype(np.int64)
	nJetsClasses_w_indices = np.column_stack((column_to_be_added, nJetsBins_cat))
	nJetsClasses_w_indices.astype(np.int64)

	nJetsClasses_w_indices_test = np.column_stack((column_to_be_added_test, nJetsBins_cat_test))
	nJetsClasses_w_indices_test.astype(np.int64)

    #print nJetsClasses_w_indices
	df_Convert = pd.DataFrame(data=nJetsClasses_w_indices[0:,1:], index=nJetsClasses_w_indices[0:,0])
	nJets_binned = df_Convert.astype('int64', copy=False)
	df_Convert_test = pd.DataFrame(data=nJetsClasses_w_indices_test[0:,1:], index=nJetsClasses_w_indices_test[0:,0])
	nJets_binned_test = df_Convert_test.astype('int64', copy=False)

	### Build the neural network ###
	from keras.models import Sequential,Model
	from keras.layers import Input,Dense,Activation,Dropout
	from keras import optimizers
	from sklearn.utils import class_weight

	# Define the architecture
	inputs = Input(shape = (train_x.shape[1], ))
	premodel = Dense(100, kernel_initializer='normal', activation='relu')(inputs)
	premodel = Dropout(0.2)(premodel)
	premodel = Dense(100, kernel_initializer='normal', activation='relu')(premodel)
	premodel = Dropout(0.2)(premodel)
	premodel = Dense(50, kernel_initializer='normal', activation='relu')(premodel)
	premodel = Dense(1, kernel_initializer='normal', activation='sigmoid')(premodel)

	model = Model(inputs=[inputs], outputs=[premodel])

	# Adversarial network architecture
	advPremodel = model(inputs)
	advPremodel = Dense(100, kernel_initializer='normal', activation='relu')(advPremodel)
	advPremodel = Dropout(0.2)(advPremodel)
	advPremodel = Dense(100, kernel_initializer='normal', activation='relu')(advPremodel)
	advPremodel = Dropout(0.2)(advPremodel)
	advPremodel = Dense(50, kernel_initializer='normal', activation='relu')(advPremodel)
	advPremodel = Dense(NJETS_BINS.size-1, kernel_initializer='normal', activation='softmax')(advPremodel)

	advmodel = Model(inputs=[inputs], outputs=[advPremodel])

	###########################################

	#### Define the loss functions ####
	def make_loss_model(c):
		def loss_model(y_true, y_pred):
			return c * K.binary_crossentropy(y_true, y_pred)
		return loss_model

	def make_loss_advmodel(c):
		def loss_advmodel(z_true, z_pred):
			return c * K.categorical_crossentropy(z_true, z_pred)
		return loss_advmodel
	############################

	opt_model = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
	model.compile(loss=[make_loss_model(c=1.0)], optimizer=opt_model, metrics=['accuracy'])

	opt_DRf = keras.optimizers.SGD(momentum=0., lr=0.001)
	DRf = Model(inputs=[inputs], outputs=[model(inputs), advmodel(inputs)])
	DRf.compile(loss=[make_loss_model(c=1.0), make_loss_advmodel(c=-lam)], optimizer=opt_DRf)

	opt_DfR = keras.optimizers.SGD(momentum=0., lr=0.001)
	DfR = Model(inputs=[inputs], outputs=[advmodel(inputs)])
	DfR.compile(loss=[make_loss_advmodel(c=1.0)], optimizer=opt_DfR)

	#Pretraining of "model"
	model.trainable = True
	advmodel.trainable = False

	numberOfEpochs = 100
	batchSize = 256
	earlystop1 = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10)

	# With sample weights
	model.fit(train_x,
						train_y,
						epochs=numberOfEpochs,
						batch_size = batchSize,
						callbacks=[earlystop1],
						validation_split=0.1,
						shuffle=True)

	print ' - first test set roc auc: ', round(roc_auc_score(test_y,model.predict(test_x)), 4)

	save_path = '/work/kimmokal/susyDNN/models/'
	save_name_preadv = 'susyDNN_preadv_model_nJets_dilepton_lambda'+str(lam)+'_'+str(mass_point)
	model.save(save_path+save_name_preadv+'.h5')

	if lam >= 0.0:
		#Pretraining of "advmodel"
		model.trainable = False
		advmodel.trainable = True

		model.compile(loss=[make_loss_model(c=1.0)], optimizer=opt_model, metrics=['accuracy'])
		DRf.compile(loss=[make_loss_model(c=1.0), make_loss_advmodel(c=-lam)], optimizer=opt_DRf)
		DfR.compile(loss=[make_loss_advmodel(c=1.0)], optimizer=opt_DfR)

		# Separate the background
		bkg_x = train_x[train_y == 0].copy()
		nJets_binned_bkg = nJets_binned[train_y == 0].copy()

		losses = {"L_f": [], "L_r": [], "L_f - L_r": []}

		# Define batch size
		batch_size = 128

		adv_numberOfEpochs = 100

		earlystop2 = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=10)
		DfR.fit(bkg_x,
					nJets_binned_bkg,
					callbacks=[earlystop2],
					epochs = adv_numberOfEpochs)

		# Adversarial training
		num_epochs = 200
		for i in range(num_epochs):
			print 'Adversarial training epoch: ', i+1

			l = DRf.evaluate(test_x, [test_y, nJets_binned_test])
			losses["L_f - L_r"].append(l[0][None][0])
			losses["L_f"].append(l[1][None][0])
			losses["L_r"].append(-l[2][None][0])
			print(losses["L_r"][-1] / lam)

			plot_losses(i, losses, lam, num_epochs, 'Losses_nJets_dilepton_lambda'+str(lam))

			#Fit "model"
			model.trainable = True
			advmodel.trainable = False

			model.compile(loss=[make_loss_model(c=1.0)], optimizer=opt_model, metrics=['accuracy'])
			DRf.compile(loss=[make_loss_model(c=1.0), make_loss_advmodel(c=-lam)], optimizer=opt_DRf)
			DfR.compile(loss=[make_loss_advmodel(c=1.0)], optimizer=opt_DfR)

			indices = np.random.permutation(len(train_x))[:batch_size]
			DRf.train_on_batch(train_x.iloc[indices], [train_y.iloc[indices], nJets_binned.iloc[indices]])

			#Fit "advmodel"
			if lam >= 0.0:
				model.trainable = False
				advmodel.trainable = True

				model.compile(loss=[make_loss_model(c=1.0)], optimizer=opt_model, metrics=['accuracy'])
				DRf.compile(loss=[make_loss_model(c=1.0), make_loss_advmodel(c=-lam)], optimizer=opt_DRf)
				DfR.compile(loss=[make_loss_advmodel(c=1.0)], optimizer=opt_DfR)

				DfR.fit(bkg_x, nJets_binned_bkg, batch_size=batch_size, epochs=1, verbose=1)

	# Save the model
	save_path = '/work/kimmokal/susyDNN/models/'
	save_name = 'susyDNN_adv_model_nJets_dilepton_lambda'+str(lam)+'_'+str(mass_point)
	model.save(save_path+save_name+'.h5')

	print ' - second test set roc auc: ', round(roc_auc_score(test_y,model.predict(test_x)), 4)
