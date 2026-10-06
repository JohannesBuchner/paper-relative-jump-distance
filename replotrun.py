import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

for filename in sys.argv[1:]:
	df = pd.read_csv(filename)

	# Extract relevant columns
	distance = df['jump-distance']
	reference_distance = df['reference-distance']
	iteration = np.arange(len(distance))

	# Plot line plot
	fig, axs = plt.subplots(2, 1, figsize=(5, 6), gridspec_kw=dict(height_ratios=[4, 1], hspace=0), sharex=True)

	axs[0].scatter(distance, iteration, label='JD', s=2)
	axs[0].plot(reference_distance, iteration, label='ref', color='k', lw=2, ls='-')

	# Plot histogram
	axs[1].hist(distance / reference_distance, bins=np.logspace(-3, 3, 40), histtype='step', label='RJD', lw=2, color='orange')
	axs[0].set_ylim(len(distance), 0)
	axs[0].set_xlim(1e-3, 3e2)
	axs[1].set_xlim(1e-3, 3e2)
	axs[0].set_xscale('log')
	axs[1].set_xscale('log')
	axs[1].set_xlabel('Distance')
	axs[0].set_ylabel('Iteration')
	axs[1].set_ylabel('Frequency')
	axs[0].legend()
	#plt.grid(True)
	plt.savefig(filename + '.pdf')
	plt.close()
