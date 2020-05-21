# from IPython import display
import matplotlib.pyplot as plt
import numpy as np

out_dir = 'plots/adversarial_plots/'

def plot_losses(i, losses, lam, num_epochs, save_name):
	# display.clear_output(wait=True)
	# display.display(plt.gcf())

	ax1 = plt.subplot(311)
	values = np.array(losses["L_f"])
	#plt.plot(range(len(values)), values, label=r"$L_f$", color="blue")
	#plt.legend(loc="upper right")
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
	#plt.plot(range(len(values)), values, label=r"$L_f - \lambda L_r$", color="red")
	#plt.legend(loc="upper right")
	plt.plot(range(len(values)), values, color="red")
	if(i==num_epochs-1):
		plt.plot(range(len(values)), values, label=r"$L_f - \lambda L_r$", color="red")
		plt.legend(loc="upper right")

        # plt.show()
		save_path=out_dir+save_name
		plt.savefig(save_path+'.png')
		plt.savefig(save_path+'.pdf')
		plt.close()

def plot_jensenshannon(i, distances, lam, num_epochs, save_name):
	# display.clear_output(wait=True)
	# display.display(plt.gcf())

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
