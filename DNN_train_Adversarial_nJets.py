# Restrict to one GPU in case there are several GPUs available
import imp
try:
    imp.find_module('setGPU')
    import setGPU
except ImportError:
    found = False

from keras.layers import Input, Dense, Activation, Dropout
from keras.models import Sequential, Model
from sklearn.utils import class_weight
from keras import optimizers
from Plots_Losses import plot_losses, plot_jensenshannon
from scipy.spatial import distance

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import keras.callbacks
import keras.backend as K

import numpy as np
import pandas as pd
import root_numpy
import root_pandas
import math
import sys

import tensorflow as tf
sess = tf.Session()
K.set_session(sess)


# Choose the lambda hyperparameter
lam = 7.
if lam < 0.:
	print 'Please choose non-negative value for lambda!'
	sys.exit()

# Train a neural network separately for each mass point
data_dir = '/work/kimmokal/susyDNN/preprocessedData/'
train_path = data_dir + 'train_set_reduced_bkg.root'
test_path = data_dir + 'test_set_reduced_bkg.root'

# Read the ROOT files for background and signal samples and put them into dataframes
train = root_pandas.read_root(train_path, 'tree')
test = root_pandas.read_root(test_path, 'tree')

# Drop the sample names and weights at this point
train = train.drop(columns=['sampleName', 'dM_Go_LSP'])
test = test.drop(columns=['sampleName', 'dM_Go_LSP'])

# Separate input features and the target
train_y = train['target']
test_y = test['target']

train_x = train.drop(columns=['target'])
test_x = test.drop(columns=['target'])

### NJETS Bins ##############
# Description: nJet>=4 && dPhi>0.5 | nJet bins=4,5,6,7,8,9,>=10 (7 bins in total)
NJETS_BINS = np.array([0., 0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 1])

nJetsData = train["nJets30Clean"].values.flatten()
nJetsData_test = test["nJets30Clean"].values.flatten()  # Test set

nJetsBins = np.digitize(nJetsData, NJETS_BINS) - 1
nJetsBins_test = np.digitize(nJetsData_test,  NJETS_BINS) - 1
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

# column_to_be_added.astype(np.int64)
nJetsClasses_w_indices = np.column_stack((column_to_be_added, nJetsBins_cat))
nJetsClasses_w_indices.astype(np.int64)

nJetsClasses_w_indices_test = np.column_stack((column_to_be_added_test, nJetsBins_cat_test))
nJetsClasses_w_indices_test.astype(np.int64)

# print nJetsClasses_w_indices
df_Convert = pd.DataFrame(data=nJetsClasses_w_indices[0:, 1:], index=nJetsClasses_w_indices[0:, 0])
nJets_binned = df_Convert.astype('int64', copy=False)
df_Convert_test = pd.DataFrame(data=nJetsClasses_w_indices_test[0:, 1:], index=nJetsClasses_w_indices_test[0:, 0])
nJets_binned_test = df_Convert_test.astype('int64', copy=False)

### Build the neural network ###

# Define the architecture
inputs = Input(shape=(train_x.shape[1], ))
premodel = Dense(100, kernel_initializer='normal', activation='tanh')(inputs)
premodel = Dropout(0.2)(premodel)
premodel = Dense(100, kernel_initializer='normal', activation='relu')(premodel)
premodel = Dropout(0.2)(premodel)
premodel = Dense(50, kernel_initializer='normal', activation='relu')(premodel)
premodel = Dense(1, kernel_initializer='normal',activation='sigmoid')(premodel)

model = Model(inputs=[inputs], outputs=[premodel])

# Adversarial network architecture
advPremodel = model(inputs)
advPremodel = Dense(100, kernel_initializer='normal',activation='relu')(advPremodel)
advPremodel = Dropout(0.2)(advPremodel)
advPremodel = Dense(100, kernel_initializer='normal',activation='relu')(advPremodel)
advPremodel = Dropout(0.2)(advPremodel)
advPremodel = Dense(50, kernel_initializer='normal',activation='relu')(advPremodel)
advPremodel = Dense(NJETS_BINS.size - 1, kernel_initializer='normal',activation='softmax')(advPremodel)

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
 ###########################


opt_model = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
model.compile(loss=[make_loss_model(c=1.0)], optimizer=opt_model, metrics=['accuracy'])

 ########## Adaptive Learning Rate for Adversary ################################


def step_decay(epoch):
	initial_lrate = 0.01
	drop = 0.5
	epochs_drop = 20.0
	lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
	return lrate

lrate = keras.callbacks.LearningRateScheduler(step_decay)

class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.losses = []
		self.lr = []

	def on_epoch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
		self.lr.append(step_decay(len(self.losses)))
		print(K.eval(self.model.optimizer.lr))

loss_history = LossHistory()
################################################################################


################### Adaptive LR for DRf ####################################
def calculate_learning_rate(num_epoch):
	initial_lrate = 0.01
	drop = 0.5
	epochs_drop = 20.0
	lrate = initial_lrate*math.pow(drop, math.floor((1+num_epoch)/epochs_drop))
	return lrate
############################################################################


opt_DRf = keras.optimizers.SGD(momentum=0.9, lr=0.01)
DRf = Model(input=[inputs], output=[model(inputs), advmodel(inputs)])
DRf.compile(loss=[make_loss_model(c=1.0), make_loss_advmodel(c=-lam)], optimizer=opt_DRf)

opt_DfR = keras.optimizers.SGD(momentum=0.9, lr=0.01)

DfR = Model(input=[inputs], output=[advmodel(inputs)])
DfR.compile(loss=[make_loss_advmodel(c=1.0)], optimizer=opt_DfR)

classWeight = class_weight.compute_class_weight('balanced', np.unique(train_y), train_y[:])

# Pretraining of "model"
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
				class_weight=classWeight,
				shuffle=True)

print ' - first test set roc auc: ', round(roc_auc_score(test_y,model.predict(test_x)), 4)

save_path = '/work/kimmokal/susyDNN/models/'
save_name_preadv = 'susyDNN_preadv_model_reduced_bkg_lambda'+str(lam)
model.save(save_path+save_name_preadv+'.h5')

# Pretraining of "advmodel"
model.trainable = False
advmodel.trainable = True

model.compile(loss=[make_loss_model(c=1.0)], optimizer=opt_model, metrics=['accuracy'])
DRf.compile(loss=[make_loss_model(c=1.0), make_loss_advmodel(c=-lam)], optimizer=opt_DRf)
DfR.compile(loss=[make_loss_advmodel(c=1.0)], optimizer=opt_DfR)

# Separate the background
bkg_x = train_x[train_y == 0].copy()
nJets_binned_bkg = nJets_binned[train_y == 0].copy()

losses = {"L_f": [], "L_r": [], "L_f - L_r": []}
js_distances = {"JS1": [], "JS2": []}

adv_numberOfEpochs = 100
earlystop2 = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=10)
DfR.fit(bkg_x,
nJets_binned_bkg,
callbacks=[earlystop2],
epochs = adv_numberOfEpochs)

# Adversarial training
batch_size = 128
num_epochs = 10
for i in range(num_epochs):
	print 'Adversarial training epoch: ', i+1
	# Fit "model"
	model.trainable = True
	advmodel.trainable = False

	model.compile(loss=[make_loss_model(c=1.0)], optimizer=opt_model, metrics=['accuracy'])
	DRf.compile(loss=[make_loss_model(c=1.0), make_loss_advmodel(c=-lam)], optimizer=opt_DRf)
	DfR.compile(loss=[make_loss_advmodel(c=1.0)], optimizer=opt_DfR)

	indices = np.random.permutation(len(train_x))[:batch_size]

	# Set learning learning for DRf according to num_epoch
	current_learning_rate = calculate_learning_rate(i)
	K.set_value(DRf.optimizer.lr, current_learning_rate)

	DRf.train_on_batch(train_x.iloc[indices], [train_y.iloc[indices], nJets_binned.iloc[indices]])
	# print("learning_rate of DRf: ", K.eval(DRf.optimizer.lr))
	# print("learning_rate of DfR: ", K.eval(DfR.optimizer.lr))

	# Fit "advmodel"
	model.trainable = False
	advmodel.trainable = True
	model.compile(loss=[make_loss_model(c=1.0)], optimizer=opt_model, metrics=['accuracy'])
	DRf.compile(loss=[make_loss_model(c=1.0), make_loss_advmodel(c=-lam)], optimizer=opt_DRf)
	DfR.compile(loss=[make_loss_advmodel(c=1.0)], optimizer=opt_DfR)
	DfR.fit(bkg_x, nJets_binned_bkg, batch_size=batch_size, epochs=1, verbose=1)

	### Calculate the JS distance for bkg DNN output distributions with nJet=([4,5], [6,7,8], [>=9]) ###
	bkg_train_njet_4to5 = bkg_x['nJets30Clean'] < 0.1
	bkg_train_njet_6to8 = (bkg_x['nJets30Clean'] > 0.1) & (bkg_x['nJets30Clean'] < 0.3)
	bkg_train_njet_geq9 = bkg_x['nJets30Clean'] > 0.3

	dnn_output_njet_4to5 = model.predict(bkg_x[ bkg_train_njet_4to5 ])
	dnn_output_njet_6to8 = model.predict(bkg_x[ bkg_train_njet_6to8 ])
	dnn_output_njet_geq9 = model.predict(bkg_x[ bkg_train_njet_geq9 ])

	bin_n = 30
	hist_dnn_output_njet_4to5, edges_4to5 = np.histogram(dnn_output_njet_4to5, bins=bin_n, range=(0,1), density=1)
	hist_dnn_output_njet_6to8, edges_6to8 = np.histogram(dnn_output_njet_6to8, bins=bin_n, range=(0,1), density=1)
	hist_dnn_output_njet_geq9, edges_geq9 = np.histogram(dnn_output_njet_geq9, bins=bin_n, range=(0,1), density=1)

	js1 = distance.jensenshannon(hist_dnn_output_njet_6to8, hist_dnn_output_njet_4to5)
	js2 = distance.jensenshannon(hist_dnn_output_njet_geq9, hist_dnn_output_njet_6to8)
	js_distances["JS1"].append(js1)
	js_distances["JS2"].append(js2)
	print 'DNN output Jensen-Shannon distance (nJet = [6,7,8] vs. nJet = [4,5]): ' + str(js1)
	print 'DNN output Jensen-Shannon distance (nJet >= 9 vs. nJet = [6,7,8]): ' + str(js2)

	### Save losses and plot
	l = DRf.evaluate(test_x, [test_y, nJets_binned_test])
	losses["L_f - L_r"].append(l[0][None][0])
	losses["L_f"].append(l[1][None][0])
	losses["L_r"].append(-l[2][None][0])
	print("Loss: " + str(losses["L_r"][-1] / lam))

	plot_losses(i, losses, lam, num_epochs, 'Losses_adversarial_reduced_bkg_lambda'+str(lam))
	plot_jensenshannon(i, js_distances, lam, num_epochs, 'JS_distance_reduced_bkg_lambda'+str(lam))

# Save the model
save_path = '/work/kimmokal/susyDNN/models/'
save_name = 'susyDNN_adv_model_reduced_bkg_lambda_'+str(lam)
model.save(save_path+save_name+'.h5')

print '- - - - - - -'
print ' - second test set roc auc: ', round(roc_auc_score(test_y,model.predict(test_x)), 4)

# Check decorrelation for test set
bkg_test_x = test_x[test_y == 0].copy()
bkg_test_njet_4to5 = bkg_test_x['nJets30Clean'] < 0.1
bkg_test_njet_6to8 = (bkg_test_x['nJets30Clean'] > 0.1) & (bkg_test_x['nJets30Clean'] < 0.3)
bkg_test_njet_geq9 = bkg_test_x['nJets30Clean'] > 0.3

dnn_output_njet_4to5 = model.predict(bkg_test_x[ bkg_test_njet_4to5 ])
dnn_output_njet_6to8 = model.predict(bkg_test_x[ bkg_test_njet_6to8 ])
dnn_output_njet_geq9 = model.predict(bkg_test_x[ bkg_test_njet_geq9 ])

bin_n = 30
hist_dnn_output_njet_4to5, edges_4to5 = np.histogram(dnn_output_njet_4to5, bins=bin_n, range=(0,1), density=1)
hist_dnn_output_njet_6to8, edges_6to8 = np.histogram(dnn_output_njet_6to8, bins=bin_n, range=(0,1), density=1)
hist_dnn_output_njet_geq9, edges_geq9 = np.histogram(dnn_output_njet_geq9, bins=bin_n, range=(0,1), density=1)

js1 = distance.jensenshannon(hist_dnn_output_njet_6to8, hist_dnn_output_njet_4to5)
js2 = distance.jensenshannon(hist_dnn_output_njet_geq9, hist_dnn_output_njet_6to8)
print 'Test set - DNN output Jensen-Shannon distance (nJet = [6,7,8] vs. nJet = [4,5]): ' + str(js1)
print 'Test set - DNN output Jensen-Shannon distance (nJet >= 9 vs. nJet = [6,7,8]): ' + str(js2)
