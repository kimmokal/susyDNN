import numpy as np
import pandas as pd
import root_numpy
import root_pandas

# Local library
# For saving purposes, the sample names are encoded in the files as numbers
import utilities as u

# Path to the working directory and the output directory
work_dir = '/work/kimmokal/susyDNN/'
out_dir = work_dir+'/preprocessedData/'
skimmed_dir = work_dir+'preprocessedData/skimmed/'

mass_point_list = ['15_10', '19_01', '19_08', '19_10', '22_01', '22_08']

for mass_point in mass_point_list:
    ### LOAD SIGNAL FILE ###
    signal_file = skimmed_dir+'evVarFriend_T1tttt_MiniAOD_'+str(mass_point)+'_skimmed.root'
    df_sig = root_pandas.read_root(signal_file)

    # Encode the sample name to a new dataframe column
    sample_name_sig = signal_file.replace(skimmed_dir+'evVarFriend_', '').replace('_skimmed.root', '') # Strip away the irrelevant file path and extension
    df_sig['sampleName'] = u.sample_encode(sample_name_sig)

    ### LOAD BACKGROUND FILES ###
    bkg_filelist = work_dir+'background_file_list.txt'
    bkg_files = [line.rstrip('\n') for line in open(bkg_filelist)]

    # Initialize the background dataframe
    df_bkg = root_pandas.read_root(bkg_files[0])

    # Encode the sample names to a new dataframe column
    # This enables even splitting of each sample and the track keeping of individual events
    sample_name_bkg = bkg_files[0].replace(skimmed_dir+'evVarFriend_', '').replace('_skimmed.root', '')
    df_bkg['sampleName'] = u.sample_encode(sample_name_bkg)

    # Remove the first sample from the list of the bkg samples as it was used for the initialization
    bkg_files = bkg_files[1:]

    # Read the rest of the background files and append them to the dataframes
    for fname in bkg_files:
        df_new = root_pandas.read_root(fname)
        sample_name = fname.replace(skimmed_dir+'evVarFriend_', '').replace('_skimmed.root', '')
        df_new['sampleName'] = u.sample_encode(sample_name)
        df_bkg = pd.concat([df_bkg, df_new])

    # Add target values to background and signal
    df_bkg['target'] = 0
    df_sig['target'] = 1

    # # Show statistics about the samples
    # print '---- Background ----'
    # print df_bkg.describe(include='all')
    # print '-------- Signal ---------'
    # print df_sig.describe(include='all')

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
    from sklearn.preprocessing import StandardScaler
    from sklearn.externals import joblib

    float_cols = ['MET', 'LT', 'HT', 'dPhi', 'Jet1_pt', 'Jet2_pt']
    discrete_cols = ['nTop_Total_Combined', 'nJets30Clean', 'nResolvedTop', 'nBCleaned_TOTAL']
    norm_cols = float_cols + discrete_cols

    # Convert norm_cols to float64 to avoid annoying warning messages from StandardScaler
    train[norm_cols] = train[norm_cols].astype(np.float64)
    test[norm_cols] = test[norm_cols].astype(np.float64)

    # Save the variable names for later usage
    # Note: not sure if it is necessary to save All_input_features
    hickle.dump(list(train), out_dir+'All_input_features_'+str(mass_point)+'.hkl')
    hickle.dump(norm_cols, out_dir+'Normalized_input_features_'+str(mass_point)+'.hkl')

    # Use the StandardScaler to calculate the mean and the standard deviation for each input feature in the training set
    normScaler = StandardScaler().fit( train[norm_cols].values )

    # Save the scaler for later use
    joblib.dump(normScaler, out_dir+'normScaler_'+mass_point+'.pkl')

    # Normalize the training and the
    train[norm_cols] = normScaler.transform(train[norm_cols].values)
    test[norm_cols] = normScaler.transform(test[norm_cols].values)

    # Save the training and test sets
    train_out=out_dir+'train_set_'+str(mass_point)+'.root'
    train.to_root(train_out, key='tree')

    test_out=out_dir+'test_set_'+str(mass_point)+'.root'
    test.to_root(test_out, key='tree')

    print 'Mass point ', mass_point, ' processed!'
