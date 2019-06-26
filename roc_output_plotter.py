#Restrict to one gpu
import imp
try:
        imp.find_module('setGPU')
        import setGPU
except ImportError:
        found = False

import pandas as pd
import numpy as np
import root_numpy
import root_pandas
import matplotlib.pyplot as plt

from keras.models import load_model
from sklearn.metrics import roc_auc_score, roc_curve, auc

# Path to working directory
work_dir='/work/kimmokal/susyDNN/'

# Paths to the output directory and the directory containing the test set
out_dir=work_dir+'plots/'
data_dir=work_dir+'preprocessedData/'

mass_point_list = ['15_10', '19_01', '19_08', '19_10', '22_01', '22_08']

# Choose the mass point from the list
mass_point = str(mass_point_list[0])

# Extract mGo and mLSP from the mass point
gno_mass_point = mass_point[0]+'.'+mass_point[1]
lsp_mass_point = mass_point[3]+'.'+mass_point[4]

# Load the test set to a dataframe
testset_path=data_dir+'test_set_' + mass_point + '.root'
df_test = root_pandas.read_root(testset_path)

# Separate the input features and the true class label
test_y = df_test['target']
test_x = df_test.drop(columns=['target', 'sampleName'])

# Load the DNN model
model_path = work_dir+'models/'+'susyDNN_model_' + mass_point + '.h5'
model = load_model(model_path)

# DNNs predictions for each event in the test set
pred_y = model.predict(test_x)

# ---------------------------------------------------#
###### DNN output plots ######
# ---------------------------------------------------#
plt.clf()
plt.figure()

# Define the binning
binning = np.arange(0.0, 1.0, 0.04)

# Separate the DNN output for signal and background events
bkg_output = pred_y[test_y==0]
sig_output = pred_y[test_y==1]

# Plot the histograms
plt.hist( bkg_output, bins=binning, alpha=0.8, label="Background", density=1 )
plt.hist( sig_output, bins=binning, alpha=0.8, label="Signal", density=1 )
plt.xlim([0.0,1.0])
plt.legend()
plt.xlabel('Output value')
plt.title(r'$m_\tilde{g} = $' + gno_mass_point + ' TeV' + r'$, m_{\tilde{\chi}^{0}_{1}} = $' + lsp_mass_point + ' TeV')
plt.savefig(out_dir + 'classifier_output_' + mass_point + '.pdf')
plt.savefig(out_dir + 'classifier_output_' + mass_point + '.png')
plt.close()

# -------------------------------------------------#
###### ROC curve plots ######
# -------------------------------------------------#
plt.clf()
plt.figure()

# Get the false positive rate and the true positive rate and the area under the ROC curve (AUC)
fpr, tpr, thresholds  = roc_curve(test_y, pred_y)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.plot(fpr, tpr, 'r', label='ROC AUC = %0.4f'% roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.legend(loc='lower right')
plt.title(r'$m_\tilde{g} = $' + gno_mass_point + ' TeV' + r'$, m_{\tilde{\chi}^{0}_{1}} = $' + lsp_mass_point + ' TeV')
plt.ylabel('Signal acceptance rate')
plt.xlabel('Background acceptance rate')
plt.savefig(out_dir + 'roc_curve_' + mass_point + '.pdf')
plt.savefig(out_dir + 'roc_curve_' + mass_point + '.png')
plt.close()
