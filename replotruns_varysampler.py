import json
import re
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Extract all input filenames
# Expected pattern: logs/problem-Nd-{sampler}-K400-nsteps10/plots/stepsampler.pdf.txt.gz
# We vary sampler, fixed problem, K and nsteps

filenames = sys.argv[1:]

# Parse each filename to extract sampler name, K and nsteps
parsed = []
for filename in filenames:
    prefix = filename.replace('/plots/stepsampler.pdf.txt.gz', '')
    m_k = re.search(r'-K(\d+)', prefix)
    m_nsteps = re.search(r'-nsteps(\d+)', prefix)
    # Extract sampler: the part between 'Nd-' and '-K', where N is one or more digits
    m_sampler = re.search(r'-\d+d-(.+?)-K\d+', prefix)
    if m_k and m_nsteps and m_sampler:
        K_val = int(m_k.group(1))
        nsteps_val = int(m_nsteps.group(1))
        sampler_name = m_sampler.group(1)
        parsed.append({
            'prefix': prefix,
            'filename': filename,
            'K': K_val,
            'nsteps': nsteps_val,
            'sampler': sampler_name,
        })
    else:
        print(f"Warning: could not parse sampler, K or nsteps from {filename}")

if not parsed:
    print("No valid input files found.")
    sys.exit(1)

# Determine output directory from first entry, stripping sampler, K, nsteps and seed parts
def get_output_dir(prefix):
    # Remove everything between the problem base and the end
    base = re.sub(r'-\d+d-[^/]+', '', prefix)
    base = re.sub(r'-nsteps\d+', '', base)
    base = re.sub(r'-seed\d+', '', base)
    base = base.rstrip('/')
    return base

output_dir = get_output_dir(parsed[0]['prefix'])
print(f"Output dir: {output_dir}")

ax = plt.figure(figsize=(5, 3)).gca()

bins = None

for i, entry in enumerate(parsed):
    prefix = entry['prefix']
    sampler_name = entry['sampler']
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

    if bins is None:
        binlo = max(0.0001, min(0.01, geom_mean_rjd / 10, np.quantile(rjd, 0.02)))
        binhi = min(100, max(4, geom_mean_rjd * 4, np.quantile(rjd, 0.98)))
        bins = np.geomspace(binlo, binhi, 60)

    counts, bins_out = np.histogram(np.clip(rjd, binlo, binhi), bins=bins)
    norm_counts = counts / counts.max() if counts.max() > 0 else counts

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color = color_cycle[i % len(color_cycle)]

    lw = 2 if i < 7 else 1
    ls = '-' if i < 7 else '--'
    label = '%s | %d%%' % (sampler_name, 100 * np.mean(distance > reference_distance))

    x_step = np.repeat(bins_out, 2)
    y_step = np.concatenate([[0], np.repeat(norm_counts, 2), [0]])
    ax.plot(x_step, y_step, lw=lw, ls=ls, color=color, label=label)

    ax.plot(geom_mean_rjd, norm_counts.max() * 1.04, 'v', color=color, ms=12, mfc='white')

ax.set_xscale('log')
ax.set_xlabel('Relative jump distance (RJD)')
ax.set_ylabel('Normalized frequency')
ax.legend(loc='upper left', title='Sampler | f(RJD>1)')

if bins is not None:
    ax.set_xlim(bins[0], bins[-1])
ax.set_ylim(0, 1.1)

K_val = parsed[0]['K']
nsteps_val = parsed[0]['nsteps']
save_path = output_dir + f'-summary-stepsampler-varySampler-K{K_val}-nsteps{nsteps_val}.pdf'
plt.savefig(save_path)
print(f"Saved: {save_path}")
plt.close()
