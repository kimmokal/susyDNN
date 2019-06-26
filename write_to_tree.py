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
import glob
import hickle

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from keras.models import load_model

# Path to the original friend trees
work_dir = '/work/kimmokal/susyDNN/'
input_dir = work_dir+'/friend_trees/'

# Directory where the new friend trees with the DNN output are saved
out_dir = '/work/data/SUSY_1lep/FRIENDS_26JUNE_DNN_OUTPUT/'

# Choose the mass point from the list of mass points
mass_point_list = ['15_10', '19_01', '19_08', '19_10', '22_01', '22_08']
mass_point = str(mass_point_list[0])

# Load the corresponding trained DNN model
model_path = work_dir+'models/'+'susyDNN_model_' + mass_point + '.h5'
dnn_model = load_model(model_path)

# List all the friend trees and process them one by one
friend_tree_list = glob.glob(input_dir+'*.root')

# # For testing purposes
# friend_tree_list = [friend_tree_list[0]]

for friend_tree in friend_tree_list:
    # Load the friend tree to a dataframe
    df = root_pandas.read_root(friend_tree)

    # Load the list of normalized variables and the StandardScaler
    load_dir = work_dir+'/preprocessedData/'

    normScaler = joblib.load(load_dir+'normScaler_'+mass_point+'.pkl')
    normalized_features = hickle.load(load_dir+'Normalized_input_features_'+str(mass_point)+'.hkl')

    # Process the friend tree to be suitable for the DNN
    df_dnn_input = df[normalized_features].copy()
    df_dnn_input[normalized_features] = normScaler.transform(df_dnn_input[normalized_features].values)

    # Get the prediction from the DNN for each event
    dnn_output = np.squeeze( dnn_model.predict(df_dnn_input) ) # squeeze to get rid of one useless dimension

    # Add the DNN output to the original friend tree
    dnn_output_col_name = 'DNN_Output_' + mass_point
    df[dnn_output_col_name] = dnn_output

    # Extract the name of the sample
    sample_name = friend_tree.replace(input_dir, '').replace('.root', '') # Get rid of the file path and extension

    # Save the new friend tree to the output directory
    friend_tree_out=out_dir+sample_name+'_DNN_Output_'+str(mass_point)+'.root'
    df.to_root(friend_tree_out, key='t')

    print 'Processed: ', sample_name
