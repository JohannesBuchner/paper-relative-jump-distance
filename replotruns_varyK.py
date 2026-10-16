import json
import re
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from solutions import corrfunnelN_logz as corrfunnel_logz

# Extract all input filenames
# Expected pattern: logs/corrfunnel-10d-cube-slice-K{40,100,400,1000,4000}-*seed42-nsteps10/plots/stepsampler.pdf.txt.gz
# We vary K (live points), fixed problem and nsteps

filenames = sys.argv[1:]

# Parse each filename to extract K (live points) and nsteps
# Example: logs/corrfunnel-10d-cube-slice-K400-seed42-nsteps10/plots/stepsampler.pdf.txt.gz

parsed = []
for filename in filenames:
    prefix = filename.replace('/plots/stepsampler.pdf.txt.gz', '')
    # Extract K (live points)
    m_k = re.search(r'-K(\d+)', prefix)
    # Extract nsteps
    m_nsteps = re.search(r'-nsteps(\d+)', prefix)
    if m_k and m_nsteps:
        K_val = int(m_k.group(1))
        nsteps_val = int(m_nsteps.group(1))
        parsed.append({
            'prefix': prefix,
            'filename': filename,
            'K': K_val,
            'nsteps': nsteps_val,
        })
    else:
        print(f"Warning: could not parse K or nsteps from {filename}")

if not parsed:
    print("No valid input files found.")
    sys.exit(1)

# Determine output directory (problem label) from first entry, stripping K and nsteps parts
# e.g. logs/corrfunnel-10d-cube-slice-K400-seed42-nsteps10 -> logs/corrfunnel-10d-cube-slice
def get_output_dir(prefix):
    # Remove -K\d+ and -nsteps\d+ and -seed\d+ suffixes to get the base problem dir
    base = re.sub(r'-K\d+', '', prefix)
    base = re.sub(r'-nsteps\d+', '', base)
    base = re.sub(r'-seed\d+', '', base)
    base = base.rstrip('/')
    return base

output_dir = get_output_dir(parsed[0]['prefix'])
print(f"Output dir: {output_dir}")

# Load one result to get nparams
info = json.load(open(parsed[0]['prefix'] + '/info/results.json'))
nparams = len(info['paramnames'])

# Sort by K
parsed.sort(key=lambda x: x['K'])

ax = plt.figure(figsize=(5, 5)).gca()
ax_inset = plt.axes([0.22, 0.55, 0.25, 0.25])
ax_inset.set_facecolor('#F5F5F5')
ax_inset.set_xlabel('Live points (K)')
ax_inset.set_title('ln(Z) convergence')

logzs = []
logzerrs = []
Ks = []
bins = None

for i, entry in enumerate(parsed):
    prefix = entry['prefix']
    K_val = entry['K']
    filename = entry['filename']

    if not os.path.exists(filename):
        print(f"Missing: {filename}")
        continue

    result = json.load(open(prefix + '/info/results.json'))

    df = pd.read_csv(filename)

    distance = df['jump-distance']
    reference_distance = df['reference-distance']
    rjd = distance / reference_distance

    geom_mean_rjd = np.exp(np.mean(np.log(rjd + 1e-3)))
    frac_rjd_gt1 = np.mean(rjd > 1)

    if bins is None:
        binlo = max(0.0001, min(0.1, geom_mean_rjd / 10, np.quantile(rjd, 0.02)))
        binhi = min(100, max(4, geom_mean_rjd * 4, np.quantile(rjd, 0.98)))
        bins = np.geomspace(binlo, binhi, 60)

    counts, bins_out = np.histogram(np.clip(rjd, binlo, binhi), bins=bins)
    norm_counts = counts / counts.max() if counts.max() > 0 else counts

    # Plot normalized histogram manually
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color = color_cycle[i % len(color_cycle)]

    lw = 2 if i < 7 else 1
    ls = '-' if i < 7 else '--'
    label = 'K=%d | %d%%' % (K_val, 100 * np.mean(distance > reference_distance))

    # Draw step histogram
    x_step = np.repeat(bins_out, 2)
    y_step = np.concatenate([[0], np.repeat(norm_counts, 2), [0]])
    line, = ax.plot(x_step, y_step, lw=lw, ls=ls, color=color, label=label)

    ax.plot(geom_mean_rjd, norm_counts.max() * 1.04, 'v', color=color, ms=12, mfc='white')
    ax.set_xscale('log')
    ax.set_xlabel('Relative jump distance (RJD)')
    ax.set_ylabel('Normalized frequency')
    ax.legend(loc='upper right', title='K | f(RJD>1)')

    logzerrs.append(result['logzerr'])
    logzs.append(result['logz'])
    Ks.append(K_val)

    ax_inset.errorbar(
        x=K_val,
        y=result['logz'],
        yerr=result['logzerr'],
        color=color,
        marker='x',
    )
    if len(Ks) >= 2:
        ax_inset.plot(Ks[-2:], logzs[-2:], color='k')

ax_inset.set_xscale('linear')

if 'beta' in output_dir or 'asymgauss' in output_dir or 'loggamma' in output_dir:
    ax_inset.axhline(0, ls=':', color='gray', alpha=0.5)
if 'eggbox' in output_dir:
    ax_inset.axhline(235.88, ls=':', color='gray', alpha=0.5)
if 'corrfunnelN' in output_dir:
    label_part = output_dir.rstrip('/').split('/')[-1]
    m = re.search(r'corrfunnelN-(\d+)d', label_part)
    if m:
        total_dim = int(m.group(1))
        ndim_arg = total_dim - 1
        true_logz = corrfunnel_logz(ndim_arg)
        ax_inset.axhline(true_logz, ls=':', color='gray', alpha=0.5)

ylo, yhi = ax_inset.get_ylim()
if yhi - ylo < 1.2:
    ax_inset.set_ylim(
        (ylo + yhi) / 2 + 0.6,
        (ylo + yhi) / 2 - 0.6
    )

if bins is not None:
    ax.set_xlim(bins[0], bins[-1])
ax.set_ylim(0, 1.5)

save_path = output_dir + '-summary-stepsampler-varyK.pdf'
plt.savefig(save_path)
print(f"Saved: {save_path}")
plt.close()
