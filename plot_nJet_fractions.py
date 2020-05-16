import numpy as np
import pandas as pd
import root_numpy
import root_pandas
import random
import matplotlib.pyplot as plt

# Local library
# For saving purposes, the sample names are encoded in the files as numbers
import utilities_parametric as u

# Path to the working directory and the output directory
work_dir = '/work/kimmokal/susyDNN/'
out_dir = work_dir+'/plots/'
file_dir = work_dir+'preprocessedData/total_background_nJet_reduced.root'

#  Load the root file
df_bkg = root_pandas.read_root(file_dir)

# Decode the sample names
df_bkg['sampleName'] = df_bkg['sampleName'].apply(lambda x: u.sample_decode(x))

for column in df_bkg.columns:
    if column not in ['nJets30Clean', 'sampleName']:
        df_bkg = df_bkg.drop(columns=column)

# Set nJets > 14 to 14
df_bkg.replace([15., 16.], 14, inplace=True)

samples = ['WJetsToLNu_HT2500toInf', 'WJetsToLNu_HT1200to2500', 'WJetsToLNu_HT800to1200', 'WJetsToLNu_HT600to800',
           'WJetsToLNu_HT400to600', 'TTJets_LO_HT2500toInf_ext', 'TTJets_LO_HT1200to2500_ext', 'TTJets_LO_HT800to1200_ext',
           'TTJets_LO_HT600to800_ext', 'TTJets_SingleLeptonFromTbar', 'TTJets_SingleLeptonFromT', 'TTJets_DiLepton']
colors = ['#99ff99', '#00ff00', '#00cc00', '#009900', '#006600', '#ff66ff', '#ff77ff', '#ff00ff', '#cc00cc', '#990099',
          '#660066', '#000033']
histograms = [df_bkg[df_bkg.sampleName == sample].nJets30Clean.to_numpy() for sample in samples]

fig, ax = plt.subplots()
fig.set_size_inches(5.5, 6)
hist_n, bins, p = ax.hist(histograms, bins=range(4,16), stacked=True, color=colors, label=samples)
ax.set_xlabel('Jet multiplicity')
ax.set_xlim([0, 15])
ax.set_ylim([3, 9*10**4])
ax.set_yscale('log')
plt.savefig(out_dir+'nJet_log_after_reduction.png')
plt.close(fig)

hist_values = hist_n[-1]
bin_weights = 1/hist_values

weight_dict = {4.: bin_weights[0], 5.: bin_weights[1], 6.: bin_weights[2], 7.: bin_weights[3], 8.: bin_weights[4],
               9.: bin_weights[5], 10.: bin_weights[6], 11.: bin_weights[7], 12.: bin_weights[8], 13.:bin_weights[9], 14.:bin_weights[10]}

hist_weights = []
for h in histograms:
    hist_weights.append([weight_dict.get(x) for x in h])

fig, ax = plt.subplots()
fig.set_size_inches(6.5, 7)
hist_n, bins, p = ax.hist(histograms, bins=range(4,16), stacked=True, color=colors, weights=hist_weights, label=samples)
ax.set_xlabel('Jet multiplicity')
ax.set_xlim([0, 15])
ax.set_ylim([0, 1.025])
ax.legend(reversed(ax.legend().legendHandles), reversed(samples), loc='upper left', framealpha=1.0)
plt.savefig(out_dir+'nJet_normed_after_reduction.png')
plt.close(fig)
