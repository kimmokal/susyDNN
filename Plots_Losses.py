import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out_dir = 'plots/adversarial_plots/'

def plot_losses(i, losses, lam, num_epochs, save_name):
	ax1 = plt.subplot(311)
	values = np.array(losses["L_f"])
	plt.plot(range(len(values)), values, color="blue")
	if(i==num_epochs-1):
		plt.plot(range(len(values)), values, label=r"$L_f$", color="blue")
		plt.legend(loc="upper right")

	ax2 = plt.subplot(312, sharex=ax1)
	values = np.array(losses["L_r"]) / lam
	plt.plot(range(len(values)), values, color="green")
	if(i==num_epochs-1):
		plt.plot(range(len(values)), values, label=r"$L_r$", color="green")
		plt.legend(loc="upper right")

	ax3 = plt.subplot(313, sharex=ax1)
	values = np.array(losses["L_f - L_r"])
	plt.plot(range(len(values)), values, color="red")
	if(i==num_epochs-1):
		plt.plot(range(len(values)), values, label=r"$L_f - \lambda L_r$", color="red")
		plt.legend(loc="upper right")

		save_path=out_dir+save_name
		plt.savefig(save_path+'.png')
		plt.savefig(save_path+'.pdf')
		plt.close()

def plot_jensenshannon(i, distances, lam, num_epochs, save_name):
	js1 = np.array(distances["JS1"])
	js2 = np.array(distances["JS2"])

	if(i==num_epochs-1):
		plt.plot(range(len(js1)), js1, label="nJet = [4,5] vs [6,7,8]", color="red")
		plt.plot(range(len(js2)), js2, label="nJet = [6,7,8] vs. [>=9]", color="blue")
		plt.ylim([0,1])
		plt.title(r"Jensen-Shannon distance ($\lambda$ = " + str(lam) + ")")
		plt.legend(loc="upper right")

        # plt.show()
		save_path=out_dir+save_name
		plt.savefig(save_path+'.png')
		plt.savefig(save_path+'.pdf')
		plt.close()


def plot_inefficiencies(i, inefficiencies_compressed, inefficiencies_uncompressed, lam, num_epochs, save_name):
	ineff_sig_compressed = np.array(inefficiencies_compressed["Signal"])
	ineff_bkg_compressed = np.array(inefficiencies_compressed["Bkg"])

	ineff_sig_uncompressed = np.array(inefficiencies_uncompressed["Signal"])
	ineff_bkg_uncompressed = np.array(inefficiencies_uncompressed["Bkg"])

	if(i==num_epochs-1):
		plt.plot(range(len(ineff_sig_compressed)), ineff_sig_compressed, label="Signal Compressed (DNN<0.8/Total Range)", color="orange")
		plt.plot(range(len(ineff_sig_uncompressed)), ineff_sig_uncompressed, label="Signal Uncompressed (DNN<0.8/Total Range)", color="green")
		plt.plot(range(len(ineff_bkg_compressed)), ineff_bkg_compressed, label="Bkg (DNN>0.8/Total Range)", color="blue")
		plt.ylim([0,1.2])
		plt.title(r"Inefficiencies ($\lambda$ = " + str(lam) + ")")
		plt.legend(loc="upper right")
		plt.axhline(y=1, linestyle='--', color='black')

		save_path=out_dir+save_name
		plt.savefig(save_path+'.png')
		plt.savefig(save_path+'.pdf')
		plt.close()
