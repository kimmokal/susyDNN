import numpy as np
import pandas as pd
import root_numpy
import root_pandas
import random

# Local library
# For saving purposes, the sample names are encoded in the files as numbers
import utilities_all_bkg as u

# Path to the working directory and the output directory
work_dir = '/work/kimmokal/susyDNN/'
out_dir = work_dir+'/preprocessedData/'
skimmed_dir = work_dir+'preprocessedData/skimmed_all_bkg_withMT/'

### LOAD SIGNAL FILES ###
signal_file_compressed = skimmed_dir+'evVarFriend_SIGNAL_Compressed_skimmed.root'
df_sig_compressed = root_pandas.read_root(signal_file_compressed)
signal_file_uncompressed = skimmed_dir+'evVarFriend_SIGNAL_Uncompressed_skimmed.root'
df_sig_uncompressed = root_pandas.read_root(signal_file_uncompressed)

# Encode the sample name to a new dataframe column
sample_name_sig_compressed = signal_file_compressed.replace(skimmed_dir+'evVarFriend_', '').replace('_skimmed.root', '') # Strip away the irrelevant file path and extension
df_sig_compressed['sampleName'] = u.sample_encode(sample_name_sig_compressed)
sample_name_sig_uncompressed = signal_file_uncompressed.replace(skimmed_dir+'evVarFriend_', '').replace('_skimmed.root', '')
df_sig_uncompressed['sampleName'] = u.sample_encode(sample_name_sig_uncompressed)

df_sig = pd.concat([df_sig_compressed, df_sig_uncompressed])

### LOAD BACKGROUND FILES ###
bkg_filelist = work_dir+'background_all_bkg_withMT_file_list.txt'
bkg_files = [line.rstrip('\n') for line in open(bkg_filelist)]

# Initialize the background dataframe
df_bkg = root_pandas.read_root(bkg_files[0])

# Encode the sample names to a new dataframe column
# This enables even splitting of each sample and the track keeping of individual events
sample_name_first = bkg_files[0].replace(skimmed_dir+'evVarFriend_', '').replace('_skimmed.root', '')
df_bkg['sampleName'] = u.sample_encode(sample_name_first)

# Remove the first sample from the list of the bkg samples as it was used for the initialization
bkg_files = bkg_files[1:]

# Read the rest of the background files and append them to the dataframes
for fname in bkg_files:
    df_new = root_pandas.read_root(fname)
    sample_name = fname.replace(skimmed_dir+'evVarFriend_', '').replace('_skimmed.root', '')
    df_new['sampleName'] = u.sample_encode(sample_name)
    df_bkg = pd.concat([df_bkg, df_new])

df_bkg = df_bkg.sample(frac=1, random_state=42).reset_index(drop=True)
df_sig = df_sig.sample(frac=1, random_state=42).reset_index(drop=True)

# Add target values to background and signal
df_bkg['target'] = 0
df_sig['target'] = 1

### SPLIT INTO TRAINING AND TEST SETS ###
# Initialize the dataframes
df_bkg_train = pd.DataFrame()
df_bkg_test = pd.DataFrame()

df_sig_train = pd.DataFrame()
df_sig_test = pd.DataFrame()

# Do the splitting on each individual sample and choose the size of the split (by default 80/20 split)
train_split_size = 0.8

for sample_name in pd.unique(df_bkg['sampleName']):
    df_sample = df_bkg[df_bkg['sampleName'] == sample_name]

    sample_size = df_sample.shape[0]
    split_point = np.ceil(sample_size*train_split_size).astype(np.int32)

    df_sample_train = df_sample.iloc[:split_point, :]
    df_sample_test = df_sample.iloc[split_point:, :]

    df_bkg_train = pd.concat([df_bkg_train, df_sample_train])
    df_bkg_test = pd.concat([df_bkg_test, df_sample_test])

for sample_name in pd.unique(df_sig['sampleName']):
    df_sample = df_sig[df_sig['sampleName'] == sample_name]

    sample_size = df_sample.shape[0]
    split_point = np.ceil(sample_size*train_split_size).astype(np.int32)

    df_sample_train = df_sample.iloc[:split_point, :]
    df_sample_test = df_sample.iloc[split_point:, :]

    df_sig_train = pd.concat([df_sig_train, df_sample_train])
    df_sig_test = pd.concat([df_sig_test, df_sample_test])

# Merge the training and test sets
train = pd.concat([df_bkg_train, df_sig_train], ignore_index=True)
test = pd.concat([df_bkg_test, df_sig_test], ignore_index=True)

# Shuffle the training set to properly mix up the samples for the DNN training
train = train.sample(frac=1, random_state=42).reset_index(drop=True)

### NORMALIZATION ###
import hickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.externals import joblib

float_cols = ['MET', 'LT', 'HT', 'MT', 'dPhi', 'Jet1_pt', 'Jet2_pt']
discrete_cols = ['nTop_Total_Combined', 'nJets30Clean', 'nResolvedTop', 'nBCleaned_TOTAL']
norm_cols = float_cols + discrete_cols

# Convert norm_cols to float64 to avoid annoying warning messages from StandardScaler
train[norm_cols] = train[norm_cols].astype(np.float64)
test[norm_cols] = test[norm_cols].astype(np.float64)

# Save the variable names for later usage
# Note: not sure if it is necessary to save All_input_features
hickle.dump(list(train), out_dir+'All_input_features_all_bkg_combined_signal_withMT.hkl')
hickle.dump(norm_cols, out_dir+'Normalized_input_features_all_bkg_combined_signal_withMT.hkl')

# Use the MinMaxScaler to rescale the variables between -1 and 1
minmaxScaler = MinMaxScaler(feature_range=(0.0001,0.9999)).fit( train[norm_cols].values )

# Save the scaler for later use
joblib.dump(minmaxScaler, out_dir+'minmaxScaler_all_bkg_combined_signal_withMT.pkl')

# Normalize
train[norm_cols] = minmaxScaler.transform(train[norm_cols].values)
test[norm_cols] = minmaxScaler.transform(test[norm_cols].values)

# Save the training and test sets
train_out=out_dir+'train_set_all_bkg_combined_signal_withMT.root'
train.to_root(train_out, key='tree')

test_out=out_dir+'test_set_all_bkg_combined_signal_withMT.root'
test.to_root(test_out, key='tree')

print 'Processed!'
