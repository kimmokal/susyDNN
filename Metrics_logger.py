import os
import pandas as pd

def metrics_to_csv(save_path, epoch, JS1, JS2, sig_uncompressed_ineff, sig_compressed_ineff, bkg_ineff, roc_aoc):
	metrics = pd.DataFrame({'epoch' : [epoch],
	                                              'js1' : [JS1],
												  'js2' : [JS2],
												  'sig_uncomp_ineff' : [sig_uncompressed_ineff],
												  'sig_comp_ineff' : [sig_compressed_ineff],
												  'bkg_ineff' : [bkg_ineff],
												  '1 - roc auc' : roc_aoc},
												  columns=['epoch', 'js1', 'js2', 'sig_uncomp_ineff', 'sig_comp_ineff', 'bkg_ineff', '1 - roc auc'])

	metrics = metrics.round(5)

	if os.path.exists(save_path):
		with open(save_path, 'a') as f:
			metrics.to_csv(f, index=False, header=False)
	else:
		with open(save_path, 'w') as f:
			metrics.to_csv(f, index=False)
