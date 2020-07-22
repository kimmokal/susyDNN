from datetime import datetime
from keras.layers import Input, Dense, Activation, Dropout, BatchNormalization
from keras.models import Sequential, Model
from sklearn.utils import class_weight
from keras import optimizers
from scipy.spatial import distance

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.callbacks
import keras.backend as K

import numpy as np
import pandas as pd
import root_numpy
import root_pandas
import sherpa
import math
import sys
import os
import argparse

import utilities_all_bkg as u
from Plots_Losses import plot_losses, plot_jensenshannon, plot_inefficiencies
from Metrics_logger import metrics_to_csv

### Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--process", required=True, help="Process number")
parser.add_argument("-g", "--gpu", required=True, help="GPU number")
parser.add_argument("-t", "--trials", required=True, help="Number of random search trials")
args = vars(parser.parse_args())
try:
    int(args["process"])
except:
    print 'Please enter a numerical value for the process number'
    sys.exit()
process_number = int(args["process"])
if process_number < 0.:
	print 'Please enter a non-negative value for the process number'
	sys.exit()

try:
    int(args["gpu"])
except:
    print 'Please enter a numerical value for the GPU number'
    sys.exit()
gpu_number = int(args["gpu"])
if gpu_number not in [0, 1, 2, 3]:
	print 'Please enter one of the available GPUs: [0, 1, 2, 3]'
	sys.exit()

try:
    int(args["trials"])
except:
    print 'Please enter a positive integer for the number of trials'
    sys.exit()
num_of_trials = int(args['trials'])
if num_of_trials < 0.:
	print 'Please enter a positive integer for the number of trials'
	sys.exit()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess = tf.Session()
K.set_session(sess)

### Paths
data_dir = '/work/kimmokal/susyDNN/preprocessedData/'
train_path = data_dir + 'train_set_all_bkg_combined_signal_withMT.root'
test_path = data_dir + 'test_set_all_bkg_combined_signal_withMT.root'
base_model_path = '/work/kimmokal/susyDNN/models/DNN_all_bkg_random_search_woNJet/'

date = str(datetime.today().strftime('%B%d'))
save_path = base_model_path+date+'_process_'+str(process_number)+'/'
os.mkdir(save_path)

# Read the ROOT files for background and signal samples and put them into dataframes
train = root_pandas.read_root(train_path, 'tree')
test = root_pandas.read_root(test_path, 'tree')

# Separate TTdilep
tt2lep_samples = [u.sample_encode('TTJets_DiLepton'), u.sample_encode('TTJets_DiLepton_ext'), u.sample_encode('TTDileptonic_MiniAOD')]
train_x_ttdilep = train[train.sampleName.isin(tt2lep_samples)].copy()

# Separate input features and the target
train_y = train['target']
test_y = test['target']

# Separate compressed and uncompressed samples for different metrics
test_compressed = test['sampleName']==u.sample_encode('SIGNAL_Compressed')
test_uncompressed = test['sampleName']==u.sample_encode('SIGNAL_Uncompressed')

test_bkg = test['target']==0
test_compressed = test[test_compressed | test_bkg]
test_y_compressed = test_compressed['target']
test_x_compressed = test_compressed.drop(columns=['nJets30Clean', 'target', 'sampleName'])

test_uncompressed = test[test_uncompressed | test_bkg]
test_y_uncompressed = test_uncompressed['target']
test_x_uncompressed = test_uncompressed.drop(columns=['nJets30Clean', 'target', 'sampleName'])
########################################################

### NJETS Bins ##############
# Description: nJet>=4 && dPhi>0.5 | nJet bins=4,5,6,7,8,9,>=10 (7 bins in total)
NJETS_BINS = np.array([0., 0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 1])

nJetsData = train["nJets30Clean"].values.flatten()
nJetsData_test = test["nJets30Clean"].values.flatten()
nJetsData_ttdilep = train_x_ttdilep["nJets30Clean"].values.flatten()

nJetsBins = np.digitize(nJetsData, NJETS_BINS) - 1
nJetsBins_test = np.digitize(nJetsData_test,  NJETS_BINS) - 1
nJetsBins_ttdilep = np.digitize(nJetsData_ttdilep,  NJETS_BINS) - 1
## to_categorical ############
nJetsBins_cat = keras.utils.to_categorical(nJetsBins)
nJetsBins_cat_test = keras.utils.to_categorical(nJetsBins_test)
nJetsBins_cat_ttdilep = keras.utils.to_categorical(nJetsBins_ttdilep)
###################################

nJetsClasses = np.zeros((len(nJetsBins), len(NJETS_BINS)))
for i in range(len(nJetsData)):
	nJetsClasses[i, nJetsBins[i]] = 1

#### Convert from np array to pandas DataFrame ###
column_to_be_added = np.arange(len(nJetsData))
column_to_be_added_test = np.arange(len(nJetsData_test))
column_to_be_added_ttdilep = np.arange(len(nJetsData_ttdilep))

# column_to_be_added.astype(np.int64)
nJetsClasses_w_indices = np.column_stack((column_to_be_added, nJetsBins_cat))
nJetsClasses_w_indices.astype(np.int64)

nJetsClasses_w_indices_test = np.column_stack((column_to_be_added_test, nJetsBins_cat_test))
nJetsClasses_w_indices_test.astype(np.int64)

nJetsClasses_w_indices_ttdilep = np.column_stack((column_to_be_added_ttdilep, nJetsBins_cat_ttdilep))
nJetsClasses_w_indices_ttdilep.astype(np.int64)

df_Convert = pd.DataFrame(data=nJetsClasses_w_indices[0:, 1:], index=nJetsClasses_w_indices[0:, 0])
nJets_binned = df_Convert.astype('int64', copy=False)
df_Convert_test = pd.DataFrame(data=nJetsClasses_w_indices_test[0:, 1:], index=nJetsClasses_w_indices_test[0:, 0])
nJets_binned_test = df_Convert_test.astype('int64', copy=False)
df_Convert_ttdilep = pd.DataFrame(data=nJetsClasses_w_indices_ttdilep[0:, 1:], index=nJetsClasses_w_indices_ttdilep[0:, 0])
nJets_binned_ttdilep = df_Convert_ttdilep.astype('int64', copy=False)

train_x = train.drop(columns=['target', 'sampleName', 'nJets30Clean'])

bkg_test = test[test_y == 0].copy()
bkg_test_x_nJet = bkg_test['nJets30Clean']
bkg_test_x = bkg_test.drop(columns=['target', 'sampleName', 'nJets30Clean'])
test_x = test.drop(columns=['target', 'sampleName', 'nJets30Clean'])

train_x_ttdilep_nJet = train_x_ttdilep['nJets30Clean']
train_x_ttdilep = train_x_ttdilep.drop(columns=['target', 'sampleName', 'nJets30Clean'])

### Hyperparameter search ###
parameters = [sherpa.Continuous('DRf_learning_rate', [0.001, 0.1], scale='log'),
                        sherpa.Continuous('DfR_learning_rate', [0.001, 0.1], scale='log'),
                        sherpa.Continuous('DRf_momentum', [0.5, 0.9]),
                        sherpa.Continuous('DfR_momentum', [0.5, 0.9]),
                        sherpa.Discrete('additional_hidden_layers_classifier', [0, 3]),
                        sherpa.Discrete('additional_hidden_layers_adversarial', [0, 3]),
                        sherpa.Ordinal('num_hidden_units_classifier', [50, 100, 150]),
                        sherpa.Ordinal('num_hidden_units_adversarial', [50, 100, 150]),
                        sherpa.Choice('first_layer_activation', ['tanh', 'no_tanh']),
                        sherpa.Continuous('dropout_rate', [0.2, 0.5]),
                        sherpa.Ordinal('adversarial_batch_size', [128, 256, 512, 1024]),
                        sherpa.Continuous('lambda', [2, 12])]

random_search = sherpa.algorithms.RandomSearch(max_num_trials=num_of_trials)
study = sherpa.Study(parameters=parameters, algorithm=random_search, lower_is_better=True, disable_dashboard=True)

trial_num = 0
for trial in study:
    K.clear_session()
    trial_num = trial_num + 1
    print '\nTrial number ' + str(trial_num) +'\n'

    model_name = 'all_bkg_random_search_woNJet_'+date+'_process_'+str(process_number)+'_trial_'+str(trial_num)

    # Write hyperparameters to file
    with open(save_path+model_name+'_hyperparameters.txt', 'w') as f:
        for p in trial.parameters:
            if type(trial.parameters[p])==np.float: f.write(str(p)+': '+str(round(trial.parameters[p], 3))+'\n')
            else: f.write(str(p)+': '+str(trial.parameters[p])+'\n')

    lam = round(trial.parameters['lambda'], 1)

    ### Build the neural network ###
    # Define the architecture
    dropout_rate = round(trial.parameters['dropout_rate'], 2)
    num_hidden_units_classifier = trial.parameters['num_hidden_units_classifier']
    num_hidden_units_adversarial = trial.parameters['num_hidden_units_adversarial']

    first_layer_activation = trial.parameters['first_layer_activation']
    first_layer_initializer = 'glorot_uniform'
    if first_layer_activation == 'no_tanh':
        first_layer_activation = 'relu'
        first_layer_initializer = 'he_uniform'

    inputs = Input(shape=(train_x.shape[1], ))
    premodel = Dense(num_hidden_units_classifier, kernel_initializer=first_layer_initializer, activation=first_layer_activation)(inputs)
    premodel = Dropout(dropout_rate)(premodel)
    for additional_layer in range(trial.parameters['additional_hidden_layers_classifier']):
        premodel = Dense(num_hidden_units_classifier, kernel_initializer='he_uniform', activation='relu')(premodel)
        premodel = Dropout(dropout_rate)(premodel)
    premodel = Dense(50, kernel_initializer='he_uniform', activation='relu')(premodel)
    premodel = Dense(1, kernel_initializer='glorot_uniform',activation='sigmoid')(premodel)

    model = Model(inputs=[inputs], outputs=[premodel])

    # Adversarial network architecture
    advPremodel = model(inputs)
    advPremodel = Dense(num_hidden_units_adversarial, kernel_initializer='he_uniform', activation='relu')(advPremodel)
    advPremodel = Dropout(dropout_rate)(advPremodel)
    for additional_layer in range(trial.parameters['additional_hidden_layers_adversarial']):
        advPremodel = Dense(num_hidden_units_adversarial, kernel_initializer='he_uniform', activation='relu')(advPremodel)
        advPremodel = Dropout(dropout_rate)(advPremodel)
    advPremodel = Dense(50, kernel_initializer='he_uniform', activation='relu')(advPremodel)
    advPremodel = Dense(NJETS_BINS.size - 1, kernel_initializer='glorot_uniform', activation='softmax')(advPremodel)

    advmodel = Model(inputs=[inputs], outputs=[advPremodel])

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

    opt_DRf = keras.optimizers.SGD(momentum=trial.parameters['DRf_momentum'], lr=trial.parameters['DRf_learning_rate'])
    DRf = Model(input=[inputs], output=[model(inputs), advmodel(inputs)])
    DRf.compile(loss=[make_loss_model(c=1.0), make_loss_advmodel(c=-lam)], optimizer=opt_DRf)

    opt_DfR = keras.optimizers.SGD(momentum=trial.parameters['DfR_momentum'], lr=trial.parameters['DfR_learning_rate'])
    DfR = Model(input=[inputs], output=[advmodel(inputs)])
    DfR.compile(loss=[make_loss_advmodel(c=1.0)], optimizer=opt_DfR)

    # Pretraining of "model"
    model.trainable = True
    advmodel.trainable = False

    preClass_numberOfEpochs = 100
    preClass_batchSize = 256

    # With sample weights
    model.fit(train_x,
              train_y,
              epochs=preClass_numberOfEpochs,
              batch_size = preClass_batchSize,
              validation_split=0.2,
              shuffle=True)

    print ' - first test set roc auc: ', round(roc_auc_score(test_y,model.predict(test_x)), 4)

    save_name_preadv = 'susyDNN_preadv_model_'+model_name+'_lambda_'+str(lam)
    model.save(save_path+save_name_preadv+'.h5')

    # Pretraining of "advmodel"
    model.trainable = False
    advmodel.trainable = True

    model.compile(loss=[make_loss_model(c=1.0)], optimizer=opt_model, metrics=['accuracy'])
    DRf.compile(loss=[make_loss_model(c=1.0), make_loss_advmodel(c=-lam)], optimizer=opt_DRf)
    DfR.compile(loss=[make_loss_advmodel(c=1.0)], optimizer=opt_DfR)

    losses = {"L_f": [], "L_r": [], "L_f - L_r": []}
    js_distances = {"JS1": [], "JS2": []}
    inefficiencies_compressed = {"Signal" : [], "Bkg" : []}
    inefficiencies_uncompressed = {"Signal" : [], "Bkg" : []}

    preAdv_numberOfEpochs = 100
    preAdv_batchSize = 256
    DfR.fit(train_x_ttdilep, nJets_binned_ttdilep, epochs = preAdv_numberOfEpochs, batch_size = preAdv_batchSize)

    # Adversarial training
    batch_size = trial.parameters['adversarial_batch_size']
    num_epochs = 200
    for i in range(num_epochs):
        print 'Adversarial training epoch: ', i+1
        # Fit "model"
        model.trainable = True
        advmodel.trainable = False

        model.compile(loss=[make_loss_model(c=1.0)], optimizer=opt_model, metrics=['accuracy'])
        DRf.compile(loss=[make_loss_model(c=1.0), make_loss_advmodel(c=-lam)], optimizer=opt_DRf)
        DfR.compile(loss=[make_loss_advmodel(c=1.0)], optimizer=opt_DfR)

        indices = np.random.permutation(len(train_x))[:batch_size]
        DRf.train_on_batch(train_x.iloc[indices], [train_y.iloc[indices], nJets_binned.iloc[indices]])

        # Fit "advmodel"
        model.trainable = False
        advmodel.trainable = True
        model.compile(loss=[make_loss_model(c=1.0)], optimizer=opt_model, metrics=['accuracy'])
        DRf.compile(loss=[make_loss_model(c=1.0), make_loss_advmodel(c=-lam)], optimizer=opt_DRf)
        DfR.compile(loss=[make_loss_advmodel(c=1.0)], optimizer=opt_DfR)
        DfR.fit(train_x_ttdilep, nJets_binned_ttdilep, batch_size=batch_size, epochs=1, verbose=1)

        ### Calculate Signal/Bkg inefficiencies
        pred_y_compressed = model.predict(test_x_compressed)
        bkg_output_compressed = pred_y_compressed[test_y_compressed==0]
        sig_output_compressed = pred_y_compressed[test_y_compressed==1]

        sig_compressed_ineff = float(len(sig_output_compressed[sig_output_compressed<0.8]))/float(len(sig_output_compressed))
        bkg_compressed_ineff = float(len(bkg_output_compressed[bkg_output_compressed>0.8]))/float(len(bkg_output_compressed))
        inefficiencies_compressed["Signal"].append(sig_compressed_ineff)
        inefficiencies_compressed["Bkg"].append(bkg_compressed_ineff)
        print "Signal inefficiency compressed: " + str(sig_compressed_ineff)

        pred_y_uncompressed = model.predict(test_x_uncompressed)
        bkg_output_uncompressed = pred_y_uncompressed[test_y_uncompressed==0]
        sig_output_uncompressed = pred_y_uncompressed[test_y_uncompressed==1]

        sig_uncompressed_ineff = float(len(sig_output_uncompressed[sig_output_uncompressed<0.8]))/float(len(sig_output_uncompressed))
        bkg_uncompressed_ineff = float(len(bkg_output_uncompressed[bkg_output_uncompressed>0.8]))/float(len(bkg_output_uncompressed))
        inefficiencies_uncompressed["Signal"].append(sig_uncompressed_ineff)
        inefficiencies_uncompressed["Bkg"].append(bkg_uncompressed_ineff)
        print "Signal inefficiency uncompressed: " + str(sig_uncompressed_ineff)
        print "Bkg inefficiency: " + str(bkg_uncompressed_ineff)

        ### Calculate the JS distance for bkg DNN output distributions with nJet=([4,5], [6,7,8], [>=9]) ###
        bkg_test_njet_4to5 = bkg_test_x_nJet < 0.1
        bkg_test_njet_6to8 = (bkg_test_x_nJet > 0.1) & (bkg_test_x_nJet < 0.3)
        bkg_test_njet_geq9 = bkg_test_x_nJet > 0.3

        dnn_output_njet_4to5 = model.predict(bkg_test_x[ bkg_test_njet_4to5 ])
        dnn_output_njet_6to8 = model.predict(bkg_test_x[ bkg_test_njet_6to8 ])
        dnn_output_njet_geq9 = model.predict(bkg_test_x[ bkg_test_njet_geq9 ])

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
        l = DRf.evaluate(test_x, [test_y, nJets_binned_test], batch_size=512)
        losses["L_f - L_r"].append(l[0][None][0])
        losses["L_f"].append(l[1][None][0])
        losses["L_r"].append(-l[2][None][0])
        print("Loss: " + str(losses["L_r"][-1] / lam))

        plot_losses(i, losses, lam, num_epochs, 'Losses_adversarial_'+model_name+'_lambda_'+str(lam))
        plot_jensenshannon(i, js_distances, lam, num_epochs, 'JS_distance_'+model_name+'_lambda_'+str(lam))
        plot_inefficiencies(i, inefficiencies_compressed, inefficiencies_uncompressed, lam, num_epochs, 'Inefficiencies_'+model_name+'_lambda'+str(lam))

        roc_aoc = 1 - round(roc_auc_score(test_y,model.predict(test_x)), 4)
        print 'ROC area over curve: ' + str(roc_aoc)

        # Save the metrics for the best models to a .csv file
        if (js1<=0.08 and js2<=0.08 and sig_uncompressed_ineff<=0.7 and sig_compressed_ineff<=0.8 and bkg_uncompressed_ineff<=0.1):
            metrics_path = save_path+model_name+'_best_metrics.txt'
            metrics_to_csv(metrics_path, i+1, js1, js2, sig_uncompressed_ineff, sig_compressed_ineff, bkg_uncompressed_ineff, roc_aoc)

        # Save the model
        model_save_name = 'susyDNN_adv_model_'+model_name+'_lambda_'+str(lam)+'_epoch_'+str(i+1)
        model.save(save_path+model_save_name+'.h5')
