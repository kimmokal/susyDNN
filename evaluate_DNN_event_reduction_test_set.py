#Restrict to one gpu
import imp
try:
        imp.find_module('setGPU')
        import setGPU
except ImportError:
        found = False

import sys
import argparse

import numpy as np
import pandas as pd
import root_pandas

import tensorflow as tf
sess = tf.Session()
import keras.backend as K
K.set_session(sess)
from keras.models import load_model
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.spatial import distance

import utilities_parametric as u

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True, help="Path to model .h5 file")
args = vars(parser.parse_args())

# Paths
data_path = '/work/kimmokal/susyDNN/preprocessedData/test_set_reduced_bkg.root'
model_path = args["model"]

# Load test set
df_test = root_pandas.read_root(data_path)
test_y = df_test['target'].copy()
test_x = df_test.copy().drop(columns=['target', 'sampleName', 'dM_Go_LSP'])

# Load the DNN model with the custom loss function
def make_loss_model(c):
	def loss_model(y_true, y_pred):
		return c * K.binary_crossentropy(y_true, y_pred)
	return loss_model

model = load_model(model_path, custom_objects={'loss_model': make_loss_model(c=1.0)})

# Check Jensen-Shannon distance for test set
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

print '\n- - - - - - - - - -'
print 'DNN output distribution Jensen-Shannon distance'
print '(nJet = [6,7,8] vs. nJet = [4,5]): ' + str(js1)
print '(nJet >= 9 vs. nJet = [6,7,8]): ' + str(js2)

print '- - - - - - - - - -'
print 'Overall test set ROC AUC: '+str(round(roc_auc_score(test_y,model.predict(test_x)), 4))

print '- - - - - - - - - -'
print 'Accuracy for individual MC samples'
for sample in sorted(df_test.sampleName.unique()):
    sample_y = df_test[df_test.sampleName == sample].copy().target
    sample_x = df_test[df_test.sampleName == sample].copy().drop(columns=['target', 'sampleName', 'dM_Go_LSP'])
    if len(sample_y) > 1:
        sample_pred = np.squeeze(model.predict(sample_x))
        sample_accuracy = 100*accuracy_score(sample_y, sample_pred.round())
        print u.sample_decode(sample)+' ('+str(len(sample_y))+' events)'+': '+str(round(sample_accuracy, 2))+'%'
print '- - - - - - - - - -'
