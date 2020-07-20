import glob
import os
import sys
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--signal", required=True, help="Choose whether compressed (c) or uncompressed (u)")
args = vars(parser.parse_args())
signal_mass = args["signal"]
if not (signal_mass == 'c' or signal_mass == 'compressed' or signal_mass == 'u' or signal_mass == 'uncompressed'):
    print 'Please enter either \'c\' for the compressed signal region or \'u\' for the uncompressed signal region '
    sys.exit()

compressed = True if (signal_mass == 'c' or signal_mass == 'compressed') else False
uncompressed = True if (signal_mass == 'u' or signal_mass == 'uncompressed') else False

### THRESHOLDS ###
signal_compressed_inefficiency_threshold = 0.2
signal_uncompressed_inefficiency_threshold = 0.1
bkg_inefficiency_thershold = 0.1
js1_threshold = 0.07
js2_threshold = 0.07
roc_threshold = 0.05
#################

model_path = "/work/kimmokal/susyDNN/models/"
best_metrics_paths = glob.glob(model_path+"DNN_all_bkg_random_search_*/*/*_best_metrics.txt")

metrics = pd.DataFrame()

for file_path in best_metrics_paths:
    add_metrics = pd.read_csv(file_path)
    file_name = os.path.basename(file_path.replace("_best_metrics.txt", ""))
    add_metrics['file'] = file_name

    if metrics.empty:
        metrics = add_metrics
    else:
        metrics = pd.concat([metrics, add_metrics])

if compressed:
    metrics = metrics[metrics['sig_comp_ineff'] < signal_compressed_inefficiency_threshold]
    metrics = metrics.sort_values('sig_comp_ineff')
if uncompressed:
    metrics = metrics[metrics['sig_uncomp_ineff'] < signal_uncompressed_inefficiency_threshold]
    metrics = metrics.sort_values('sig_uncomp_ineff')
metrics = metrics[metrics['bkg_ineff'] < bkg_inefficiency_thershold]
metrics = metrics[metrics['js1'] < js1_threshold]
metrics = metrics[metrics['js2'] < js2_threshold]
metrics = metrics[metrics['1 - roc auc'] < roc_threshold]

# Print the best models
num_models = 3
if metrics.shape[0] < num_models:
    num_models = metrics.shape[0]

print "PRINTING TOP " + str(num_models) + " MODELS\n"
for i in range(num_models):
    print str(i+1) + ". MODEL (" + str(metrics['file'].values[i]) + ")"
    print "Epoch: " + str(metrics['epoch'].values[i])
    if compressed:
        print "Signal inefficiency (compressed): " + str(round(metrics['sig_comp_ineff'].values[i], 4))
    if uncompressed:
        print "Signal inefficiency (uncompressed): " + str(round(metrics['sig_uncomp_ineff'].values[i], 4))
    print "Background inefficiency: " + str(round(metrics['bkg_ineff'].values[i], 4))
    print "DNN output Jensen-Shannon distance (nJet = [6,7,8] vs. nJet = [4,5]): " + str(round(metrics['js1'].values[i], 4))
    print "DNN output Jensen-Shannon distance (nJet >= 9 vs. nJet = [6,7,8]): " + str(round(metrics['js2'].values[i], 4))
    print "ROC AUC: " + str(round(1 - metrics['1 - roc auc'].values[i], 4))
    print "\n"
