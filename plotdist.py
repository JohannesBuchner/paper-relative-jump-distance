import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np


def extract_nsteps_from_path(filepath):
    match = re.search(r'nsteps(\d+)', filepath)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract nsteps from path: {filepath}")


def read_results_json(log_file):
    # log_file is something like logs/eggbox-2d-cube-slice-K400-seed42-nsteps8/stepsampler.log
    # results.json is at logs/eggbox-2d-cube-slice-K400-seed42-nsteps8/info/results.json
    log_dir = os.path.dirname(log_file)
    results_path = os.path.join(log_dir, 'info', 'results.json')
    with open(results_path) as f:
        content = f.read()
    # Handle Infinity values
    content = content.replace('Infinity', 'null')
    data = json.loads(content)
    logz = data['logz']
    logzerr = data['logzerr']
    return logz, logzerr


def read_stepsampler_log(log_file, nsteps):
    # The log file columns according to finalize_chain:
    # Lmin, ustart (ndim), ufinal (ndim), tstart (ndim), tfinal (ndim),
    # nsteps, maxradiussq**0.5, mean_pair_distance, iLstart, iLfinal, itstart, itfinal
    # We need jump-distance and reference-distance
    # jump-distance = distance in t-space between tstart and tfinal = sqrt(sum((tstart - tfinal)^2))
    # reference-distance = maxradiussq**0.5
    # But we don't know ndim from the file alone.
    # We can infer ndim from the number of columns:
    # total cols = 1 + ndim + ndim + ndim + ndim + 6 = 1 + 4*ndim + 6
    # so ndim = (ncols - 7) / 4

    data = np.loadtxt(log_file)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if len(data) == 0:
        return None, None

    ncols = data.shape[1]
    # ncols = 1 + 4*ndim + 6
    ndim = (ncols - 7) // 4

    # columns:
    # 0: Lmin
    # 1..ndim: ustart
    # 1+ndim..1+2*ndim: ufinal
    # 1+2*ndim..1+3*ndim: tstart
    # 1+3*ndim..1+4*ndim: tfinal
    # 1+4*ndim: nsteps
    # 1+4*ndim+1: maxradiussq**0.5  (reference-distance)
    # 1+4*ndim+2: mean_pair_distance
    # 1+4*ndim+3: iLstart
    # 1+4*ndim+4: iLfinal
    # 1+4*ndim+5: itstart
    # 1+4*ndim+6: itfinal

    tstart_cols = slice(1 + 2 * ndim, 1 + 3 * ndim)
    tfinal_cols = slice(1 + 3 * ndim, 1 + 4 * ndim)
    ref_col = 1 + 4 * ndim + 1

    tstart = data[:, tstart_cols]
    tfinal = data[:, tfinal_cols]
    jump_distances = np.sqrt(((tstart - tfinal) ** 2).sum(axis=1))
    reference_distances = data[:, ref_col]

    return jump_distances, reference_distances


def main():
    log_files = sys.argv[1:]
    if not log_files:
        print("Usage: python solution.py <stepsampler.log> [<stepsampler.log> ...]")
        sys.exit(1)

    fig, ax = plt.subplots()

    for log_file in log_files:
        nsteps = extract_nsteps_from_path(log_file)
        logz, logzerr = read_results_json(log_file)
        jump_distances, reference_distances = read_stepsampler_log(log_file, nsteps)

        if jump_distances is None or len(jump_distances) == 0:
            print(f"No data found in {log_file}")
            continue

        rel = jump_distances / reference_distances
        log_rel = np.log10(rel + 1e-10)

        label = f'nsteps={nsteps}, logZ={logz:.2f}±{logzerr:.2f}'
        ax.hist(log_rel, histtype='step', bins='auto', label=label)

    ax.set_xlabel('$\\log_{10}$(relative step distance)')
    ax.set_ylabel('Frequency')
    ax.legend(loc='best')

    output_filename = 'stepsampler-jumphist.pdf'
    fig.savefig(output_filename)
    plt.close(fig)
    print(f"Saved plot to {output_filename}")


if __name__ == '__main__':
    main()
