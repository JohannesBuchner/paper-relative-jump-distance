import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# extract prefix from systematiclogs/ligo/ultranest-calibrator/-nsteps7/plots/stepsampler.pdf.txt.gz
prefixes = {
    filename.replace('/plots/stepsampler.pdf.txt.gz', '')
        for filename in sys.argv[1:]}
completed = set()

for example_run_prefix in prefixes:
    output_dir = example_run_prefix.split('-nsteps', 1)[0]
    if output_dir in completed:
        continue
    print(output_dir)
    info = json.load(open(example_run_prefix + '/info/results.json'))
    nparams = len(info['paramnames'])

    # plot arrangement?
    # log(z) inset top left
    # histogram of relative step samples
    ax = plt.figure(figsize=(5, 5)).gca()
    ax_inset = plt.axes([0.22, 0.55, 0.25, 0.25])
    ax_inset.set_facecolor('#F5F5F5')
    ax_inset.set_xlabel('Number of steps')
    ax_inset.set_title('ln(Z) convergence')
    logzs = []
    logzerrs = []
    for K in range(0, 100):
        nsteps = 2**K * nparams
        run_prefix = output_dir + ('-nsteps%d/' % nsteps)
        if not os.path.exists(run_prefix + '/info/results.json'):
            break
        result = json.load(open(run_prefix + '/info/results.json'))
        filename = run_prefix + '/plots/stepsampler.pdf.txt.gz'

        df = pd.read_csv(filename)

        # Extract relevant columns
        distance = df['jump-distance']
        reference_distance = df['reference-distance']
        N, bins, P = ax.hist(distance / reference_distance, bins=np.logspace(-3, 2.2, 100),
            histtype='step', lw=2 if K < 7 else 1, ls='-' if K < 7 else '--',
            label='%d | %d%%' % (nsteps, 100 * np.mean(distance > reference_distance)), )
        # plot mean
        color = P[0].get_edgecolor()
        ax.plot(np.exp(np.mean(np.log(distance / reference_distance + 1e-3))), N.max() * 1.04, 
            'v', color=color, ms=12, mfc='white')
        ax.set_xscale('log')
        ax.set_xlabel('Relative jump distance (RJD)')
        ax.set_ylabel('Frequency')
        ax.legend(loc='upper right', title='steps | f(RJD>1)')
        
        logzerrs.append(result['logzerr'])
        logzs.append(result['logz'])
        # log(Z) convergence, use colors
        ax_inset.errorbar(
            x=nsteps,
            y=result['logz'],
            yerr=result['logzerr'],
            color=color,
        )
        if len(logzs) >= 2:
            ax_inset.plot([nsteps // 2, nsteps], logzs[-2:], color='k')
    ylo, yhi = ax_inset.get_ylim()
    if yhi - ylo < 1.2:
        ax_inset.set_ylim(
            (ylo + yhi) / 2 + 0.6,
            (ylo + yhi) / 2 - 0.6
            )
    if nparams >= 10:
        ax.set_xlim(0.01, 30)
    plt.savefig(output_dir + 'summary-stepsampler.pdf')
    plt.close()
    completed.add(output_dir)
