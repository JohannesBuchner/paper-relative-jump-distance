import json
import os
import sys
import pandas as pd
import corner
import numpy as np
import matplotlib.pyplot as plt

fig = None
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for filename, color in zip(sys.argv[1:], colors):
    run_prefix = filename.replace('/chains/equal_weighted_post.txt', '')
    info = json.load(open(run_prefix + '/info/results.json'))
    posterior = pd.read_csv(filename, delimiter=' ')
    fig = corner.corner(posterior, #labels=info['paramnames'], 
        plot_datapoints=False,
        plot_density=False,
        plot_contours=True,
        fig=fig,
        levels=1 - np.exp([-1, -2, -3]),
        #no_fill_contours=True,
        color=color,
    )
plt.savefig(run_prefix + '/chains/corner.pdf')
