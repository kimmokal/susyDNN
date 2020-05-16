import numpy as np
import pandas as pd
import root_numpy
import root_pandas
import random

# Local library
# For saving purposes, the sample names are encoded in the files as numbers
import utilities_parametric as u

# Path to the working directory and the output directory
work_dir = '/work/kimmokal/susyDNN/'
out_dir = work_dir+'/preprocessedData/'
skimmed_dir = work_dir+'preprocessedData/skimmed_parametric_all_bkg/'

### LOAD SIGNAL FILE ###
signal_file = skimmed_dir+'evVarFriend_total_SIGNAL_skimmed.root'
df_sig = root_pandas.read_root(signal_file)

# Encode the sample name to a new dataframe column
sample_name_sig = signal_file.replace(skimmed_dir+'evVarFriend_', '').replace('_skimmed.root', '') # Strip away the irrelevant file path and extension
df_sig['sampleName'] = u.sample_encode(sample_name_sig)

### LOAD BACKGROUND FILES ###
bkg_filelist = work_dir+'background_reduced_file_list.txt'
bkg_files = [line.rstrip('\n') for line in open(bkg_filelist)]

# Initialize the background dataframe
df_bkg = root_pandas.read_root(bkg_files[0])

# Encode the sample names to a new dataframe column
# This enables even splitting of each sample and the track keeping of individual events
sample_name_first = bkg_files[0].replace(skimmed_dir+'evVarFriend_', '').replace('_skimmed.root', '')
if (sample_name_first == 'TTJets_DiLepton_ext') or (sample_name_first == 'TTDileptonic_MiniAOD'):
    sample_name_first = 'TTJets_DiLepton'
df_bkg['sampleName'] = sample_name_first

# Remove the first sample from the list of the bkg samples as it was used for the initialization
bkg_files = bkg_files[1:]

# Read the rest of the background files and append them to the dataframes
for fname in bkg_files:
    df_new = root_pandas.read_root(fname)
    sample_name = fname.replace(skimmed_dir+'evVarFriend_', '').replace('_skimmed.root', '')
    if (sample_name == 'TTJets_DiLepton_ext') or (sample_name == 'TTDileptonic_MiniAOD'):
        sample_name = 'TTJets_DiLepton'
    df_new['sampleName'] = sample_name
    df_bkg = pd.concat([df_bkg, df_new])

# Load sample fractions for different nJet values
nJet_fractions = pd.read_csv(work_dir+'njet_fractions.csv')

df_bkg_reduced = pd.DataFrame()

for njet in range(4,15):
    # print '- - - - - - - - - - - - - - - - - - - - - '
    # print 'nJet: '+str(njet)
    if njet == 14:
        dilepton_events = df_bkg[(df_bkg['nJets30Clean']>=njet) & (df_bkg['sampleName']=='TTJets_DiLepton')].copy()
    else:
        dilepton_events = df_bkg[(df_bkg['nJets30Clean']==njet) & (df_bkg['sampleName']=='TTJets_DiLepton')].copy()

    if df_bkg_reduced.empty:
        df_bkg_reduced = dilepton_events
    else:
        df_bkg_reduced = pd.concat([df_bkg_reduced, dilepton_events], ignore_index=True)

    dilepton_fraction = nJet_fractions[nJet_fractions.nJet==njet]['TTJets_DiLepton'].values[0]
    total_events = round(dilepton_events.shape[0] / dilepton_fraction)

    for sample in df_bkg.sampleName.unique():
        if sample == 'TTJets_DiLepton':
            continue
        else:
            if njet == 14:
                sample_events = df_bkg[(df_bkg['nJets30Clean']>=njet) & (df_bkg['sampleName']==sample)].copy()
            else:
                sample_events = df_bkg[(df_bkg['nJets30Clean']==njet) & (df_bkg['sampleName']==sample)].copy()
            sample_fraction = nJet_fractions[nJet_fractions.nJet==njet][sample].values[0]
            sample_n_before_reduction = sample_events.shape[0]
            sample_n_after_reduction = int(round(total_events * sample_fraction))

            # print sample
            # print 'nEvents before reduction: ' + str(sample_n_before_reduction) + ', after reduction: ' + str(sample_n_after_reduction)

            if sample_n_after_reduction < sample_n_before_reduction:
                n_removed = sample_n_before_reduction - sample_n_after_reduction
                drop_indices = random.sample(sample_events.index, n_removed)
                sample_events = sample_events.drop(drop_indices)
            df_bkg_reduced = pd.concat([df_bkg_reduced, sample_events], ignore_index=True)

# print '- - - - - - - - - - - - - - - - - - - - - '
# total_before = df_bkg.shape[0]
# total_after = df_bkg_reduced.shape[0]
# print "Total number of events before reduction: " + str(total_before)
# print "Total number of events after reduction: " + str(total_after)
# print 'Dropped ' + str(total_before - total_after) + ' events'

# Encode the sample names
df_bkg_reduced['sampleName'] = df_bkg_reduced['sampleName'].apply(lambda x: u.sample_encode(x))

# # Save the reduced bkg to root file
# root_out=out_dir+'background_nJet_reduced.root'
# df_bkg_reduced.to_root(root_out, key='tree')

# Add target values to background and signal
df_bkg_reduced['target'] = 0
df_sig['target'] = 1

### SPLIT INTO TRAINING AND TEST SETS ###
# Initialize the dataframes
df_bkg_train = pd.DataFrame()
df_bkg_test = pd.DataFrame()

df_sig_train = pd.DataFrame()
df_sig_test = pd.DataFrame()

# Do the splitting on each individual sample and choose the size of the split (by default 80/20 split)
train_split_size = 0.8

for sample_name in pd.unique(df_bkg_reduced['sampleName']):
    df_sample = df_bkg_reduced[df_bkg_reduced['sampleName'] == sample_name]

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

float_cols = ['MET', 'LT', 'HT', 'dPhi', 'Jet1_pt', 'Jet2_pt']
discrete_cols = ['nTop_Total_Combined', 'nJets30Clean', 'nResolvedTop', 'nBCleaned_TOTAL']
norm_cols = float_cols + discrete_cols

# Convert norm_cols to float64 to avoid annoying warning messages from StandardScaler
train[norm_cols] = train[norm_cols].astype(np.float64)
test[norm_cols] = test[norm_cols].astype(np.float64)

# Save the variable names for later usage
# Note: not sure if it is necessary to save All_input_features
hickle.dump(list(train), out_dir+'All_input_features_reduced_bkg.hkl')
hickle.dump(norm_cols, out_dir+'Normalized_input_features_reduced_bkg.hkl')

# Use the MinMaxScaler to rescale the variables between -1 and 1
minmaxScaler = MinMaxScaler(feature_range=(0.0001,0.9999)).fit( train[norm_cols].values )

# Save the scaler for later use
joblib.dump(minmaxScaler, out_dir+'minmaxScaler_dilepton_reduced_bkg.pkl')

# Normalize
train[norm_cols] = minmaxScaler.transform(train[norm_cols].values)
test[norm_cols] = minmaxScaler.transform(test[norm_cols].values)

# Save the training and test sets
train_out=out_dir+'train_set_reduced_bkg.root'
train.to_root(train_out, key='tree')

test_out=out_dir+'test_set_reduced_bkg.root'
test.to_root(test_out, key='tree')

print 'Processed!'
