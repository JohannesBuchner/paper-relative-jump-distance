import json
import re
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from solutions import corrfunnelN_logz as corrfunnel_logz, box_logz

# extract prefix from systematiclogs/ligo/ultranest-calibrator/-nsteps7/plots/stepsampler.pdf.txt.gz
# Preserve order from sys.argv[1:] using a list and seen set
seen_prefixes = set()
prefixes = []
for filename in sys.argv[1:]:
    p = filename.replace('/plots/stepsampler.pdf.txt.gz', '')
    if p not in seen_prefixes:
        seen_prefixes.add(p)
        prefixes.append(p)

# Preserve order for output labels as well
seen_output_dirs = set()
output_dir_order = []
for p in prefixes:
    od = p.split('-nsteps', 1)[0]
    if od not in seen_output_dirs:
        seen_output_dirs.add(od)
        output_dir_order.append(od)

completed = set()
used_colors = set()


bins = None
# Data storage for combined plots
combined_data = {}  # key: output_label, value: list of dicts
# Keep insertion order for combined_data labels
combined_data_order = []

# Storage for LaTeX table rows
latex_table_rows = []

for example_run_prefix in prefixes:
    output_dir = example_run_prefix.split('-nsteps', 1)[0]
    if output_dir in completed:
        continue
    info = json.load(open(example_run_prefix + '/info/results.json'))
    nparams = len(info['paramnames'])

    # Determine true logZ for this problem (if available)
    true_logz = None
    label_part = output_dir.rstrip('/').split('/')[-1]
    if 'beta' in output_dir or 'asymgauss' in output_dir or 'loggamma' in output_dir:
        true_logz = 0.0
    if 'eggbox' in output_dir:
        true_logz = 235.88
    m_box = re.search(r'box-(\d+)d', label_part)
    if m_box:
        true_logz = box_logz(int(m_box.group(1)))
    m_cfN = re.search(r'corrfunnelN-(\d+)d', label_part)
    if m_cfN:
        total_dim = int(m_cfN.group(1))
        ndim_arg = total_dim - 1
        true_logz = corrfunnel_logz(ndim_arg)

    # Determine a short problem label for the table
    problem_label = label_part

    # Determine number of live points from info if available
    nlive = info.get('nlive', info.get('num_live_points', None))

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
    problem_runs = []
    bins = None
    for K in range(-1, 100):
        nsteps = int(2**K * nparams)
        run_prefix = output_dir + ('-nsteps%d/' % nsteps)
        if not os.path.exists(run_prefix + '/info/results.json'):
            if nsteps < nparams * 2:
                print("not enough steps were run", output_dir)
            if K > 0:
                break
            else:
                continue
        print(output_dir, nsteps)
        result = json.load(open(run_prefix + '/info/results.json'))
        filename = run_prefix + '/plots/stepsampler.pdf.txt.gz'

        df = pd.read_csv(filename)

        # Extract relevant columns
        distance = df['jump-distance']
        reference_distance = df['reference-distance']
        rjd = distance / reference_distance
        # Compute summary stats for combined plots
        geom_mean_rjd = np.exp(np.mean(np.log(rjd + 1e-3)))
        frac_rjd_gt1 = np.mean(rjd > 1)

        if bins is None:
            binlo_max = float(os.environ.get('BINLO_MAX', '0.1'))
            binlo_min = float(os.environ.get('BINLO_MIN', '0.0001'))
            binhi_min = float(os.environ.get('BINHI_MIN', '4'))
            binhi_max = float(os.environ.get('BINHI_MAX', '100'))
            nbins = int(os.environ.get('NBINS', '60'))
            binlo = max(binlo_min, min(binlo_max, geom_mean_rjd / 10, np.quantile(rjd, 0.02)))
            binhi = min(binhi_max, max(binhi_min, geom_mean_rjd * 4, np.quantile(rjd, 0.98)))
            bins = np.geomspace(binlo, binhi, nbins)
            # print('bin range:', binlo, binhi, binlo_min, binlo_max)

        counts, _ = np.histogram(np.clip(rjd, binlo, binhi), bins=bins)
        max_count = counts.max() if counts.max() > 0 else 1
        norm_counts = counts / max_count * 0.85

        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        lw = 2 if K < 7 else 1
        ls = '-' if K < 7 else '--'
        line, = ax.step(bins[:-1], norm_counts, where='post', lw=lw, ls=ls,
            label='%d | %d%%' % (nsteps, 100 * np.mean(distance > reference_distance)))
        color = line.get_color()

        idx = np.searchsorted(bins, geom_mean_rjd) - 1
        idx = np.clip(idx, 0, len(norm_counts) - 1)
        # ax.plot(geom_mean_rjd, norm_counts[idx] * 1.04, 'v', color=color, ms=8, mfc='white')
        ax.plot(geom_mean_rjd, 0.90, 'v', color=color, ms=8, mfc='white')
        
        #N, bins, P = ax.hist(
        #    np.clip(rjd, binlo, binhi),
        #    bins=bins,
        #    histtype='step', lw=2 if K < 7 else 1, ls='-' if K < 7 else '--',
        #    label='%d | %d%%' % (nsteps, 100 * np.mean(distance > reference_distance))
        #)
        ## plot mean
        #color = P[0].get_edgecolor()
        # for asymgauss, loggamma, we know the analytic value, lnZ=0.
        #ax.plot(geom_mean_rjd, N.max() * 1.04, 'v', color=color, ms=12, mfc='white')
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
            marker='x',
        )
        if len(logzs) >= 2:
            ax_inset.plot([nsteps // 2, nsteps], logzs[-2:], color='k')

        problem_runs.append({
            'nsteps': nsteps,
            'logz': result['logz'],
            'logzerr': result['logzerr'],
            'geom_mean_rjd': geom_mean_rjd,
            'frac_rjd_gt1': frac_rjd_gt1,
        })

        # Collect info for LaTeX table row
        # nlive from this result if available
        run_nlive = result.get('nlive', result.get('num_live_points', nlive))

        # Format true_logz string
        if true_logz is not None:
            true_logz_str = f'{true_logz:.2f}'
        else:
            true_logz_str = '---'

        latex_table_rows.append({
            'problem_label': problem_label,
            'ndim': nparams,
            'nlive': run_nlive,
            'nsteps': nsteps,
            'geom_mean_rjd': geom_mean_rjd,
            'frac_rjd_gt1': frac_rjd_gt1,
            'logz': result['logz'],
            'logzerr': result['logzerr'],
            'true_logz': true_logz,
            'true_logz_str': true_logz_str,
        })

    if 'beta' in output_dir or 'asymgauss' in output_dir or 'loggamma' in output_dir:
        ax_inset.axhline(0, ls=':', color='gray', alpha=0.5)
    if 'eggbox' in output_dir:
        ax_inset.axhline(235.88, ls=':', color='gray', alpha=0.5)
    if 'box' in output_dir:
        # Determine the ndim argument for corrfunnel_logz from the output_dir label
        # e.g. 'corrfunnel-2d' -> ndim=1, 'corrfunnel-10d' -> ndim=9, 'corrfunnel-50d' -> ndim=49
        label_part = output_dir.rstrip('/').split('/')[-1]
        # Try to extract the dimension number from the label
        m = re.search(r'box-(\d+)d', label_part)
        if m:
            true_logz_val = box_logz(int(m.group(1)))
            print("true_logz:", true_logz_val)
            ax_inset.axhline(true_logz_val, ls=':', color='gray', alpha=0.5)
    if 'corrfunnelN' in output_dir:
        # Determine the ndim argument for corrfunnel_logz from the output_dir label
        # e.g. 'corrfunnel-2d' -> ndim=1, 'corrfunnel-10d' -> ndim=9, 'corrfunnel-50d' -> ndim=49
        label_part = output_dir.rstrip('/').split('/')[-1]
        # Try to extract the dimension number from the label
        m = re.search(r'corrfunnelN-(\d+)d', label_part)
        if m:
            total_dim = int(m.group(1))
            ndim_arg = total_dim - 1
            true_logz_val = corrfunnel_logz(ndim_arg)
            ax_inset.axhline(true_logz_val, ls=':', color='gray', alpha=0.5)

    ylo, yhi = ax_inset.get_ylim()
    if yhi - ylo < 1.2:
        ax_inset.set_ylim(
            (ylo + yhi) / 2 + 0.6,
            (ylo + yhi) / 2 - 0.6
            )
    if nparams >= 10 or True and bins is not None:
        ax.set_xlim(bins[0], bins[-1])
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(0, yhi * 1.5)
    plt.savefig(output_dir + 'summary-stepsampler.pdf')
    plt.close()
    completed.add(output_dir)

    if problem_runs:
        output_label = '-'.join(output_dir.replace('logs/', '').replace('logs-new/', '').replace('logs-prev/', '').split('-')[:2])
        if 'ligo' in output_label:
            output_label = 'ligo'
        if output_label not in combined_data:
            combined_data_order.append(output_label)
        combined_data[output_label] = problem_runs


# =============================================================================
# LaTeX table output
# =============================================================================
def _bold(s):
    return r'\textbf{' + s + r'}'


latex_lines = []
latex_lines.append(r"\begin{table}[ht]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{Summary of runs: problem label, dimensionality ($d$), number of live points ($N_\mathrm{live}$), number of steps, geometric mean RJD, fraction with RJD$>1$, evidence estimate $\ln Z \pm \sigma$, and true value (if available). Bold geometric mean RJD indicates $<1$; bold $f(\mathrm{RJD}>1)$ indicates $<0.5$.}")
latex_lines.append(r"\label{tab:summary}")
latex_lines.append(r"\resizebox{\textwidth}{!}{%")
latex_lines.append(r"\begin{tabular}{lrrrrrrrl}")
latex_lines.append(r"\hline")
latex_lines.append(r"Name & $d$ & $N_\mathrm{live}$ & Steps & <RJD> & $f(\mathrm{RJD}>1)$ & $\ln Z$ & $\sigma_{\ln Z}$ & True $\ln Z$ \\")
latex_lines.append(r"\hline")

for row in latex_table_rows:
    problem_label = row['problem_label'].replace('_', r'\_')
    ndim = row['ndim']
    nlive = row['nlive']
    nlive_str = str(nlive) if nlive is not None else '---'
    nsteps = row['nsteps']
    geom_mean_rjd = row['geom_mean_rjd']
    frac_rjd_gt1 = row['frac_rjd_gt1']
    logz = row['logz']
    logzerr = row['logzerr']
    true_logz_str = row['true_logz_str']

    geom_str = f'{geom_mean_rjd:.3f}'
    if geom_mean_rjd < 1.0:
        geom_str = _bold(geom_str)

    frac_str = f'{frac_rjd_gt1:.3f}'
    if frac_rjd_gt1 < 0.5:
        frac_str = _bold(frac_str)

    logz_str = f'{logz:.2f}'
    logzerr_str = f'{logzerr:.2f}'

    latex_lines.append(f'{problem_label} & {ndim} & {nlive_str} & {nsteps} & {geom_str} & {frac_str} & {logz_str} & {logzerr_str} & {true_logz_str} \\\\')

latex_lines.append(r"\hline")
latex_lines.append(r"\end{tabular}}")
latex_lines.append(r"\end{table}")

with open('runsummary.tex', 'w') as f:
    f.write('\n'.join(latex_lines) + '\n')


lnZmax = 5.5
# Combined plots
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Boolean flag: if True, delta-logZ is expressed relative to the error bar
for DELTA_LOGZ_RELATIVE_TO_ERRORBAR in False, True:

    # Compute delta-logZ relative to the last entry for each problem
    # If DELTA_LOGZ_RELATIVE_TO_ERRORBAR is True, divide delta-logZ by the error bar of each run.
    # Plot 1: geometric mean RJD vs delta-logZ
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    # Plot 2: frac(RJD>1) vs delta-logZ
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    if DELTA_LOGZ_RELATIVE_TO_ERRORBAR:
        ax1.set_xlabel(r'$(\log Z - \log Z_\mathrm{final}) / \sigma$ (from sequence of runs)')
        ax2.set_xlabel(r'$(\log Z - \log Z_\mathrm{final}) / \sigma$ (from sequence of runs)')
    else:
        ax1.set_xlabel(r'$\log Z - \log Z_\mathrm{final}$ (from sequence of runs)')
        ax2.set_xlabel(r'$\log Z - \log Z_\mathrm{final}$ (from sequence of runs)')
    ax1.set_ylabel('Geometric mean RJD (single-run)')
    ax2.set_ylabel('frac(RJD>1) (single-run)')

    # Add vertical gray shaded areas
    # For ax1 (geom mean RJD): threshold matching the convergence criterion plotted
    # For ax2 (frac RJD>1): threshold matching the convergence criterion plotted
    if DELTA_LOGZ_RELATIVE_TO_ERRORBAR:
        ax1_vspan_lo, ax1_vspan_hi = -2.0, 2.0
        ax2_vspan_lo, ax2_vspan_hi = -2.0, 2.0
    else:
        ax1_vspan_lo, ax1_vspan_hi = -0.5, 0.5
        ax2_vspan_lo, ax2_vspan_hi = -0.5, 0.5

    ax1.axvline(ax1_vspan_lo, ls=':', color='gray', alpha=0.7, zorder=0)
    ax1.axvline(ax1_vspan_hi, ls=':', color='gray', alpha=0.7, zorder=0)
    ax2.axvline(ax2_vspan_hi, ls=':', color='gray', alpha=0.7, zorder=0)
    ax2.axvline(ax2_vspan_lo, ls=':', color='gray', alpha=0.7, zorder=0)

    # Add dotted grey horizontal lines
    # ax2: frac(RJD>1) = 0.5
    ax2.axhline(0.5, ls=':', color='gray', alpha=0.7, zorder=0)
    # ax1: geom mean RJD = 1
    ax1.axhline(1.0, ls=':', color='gray', alpha=0.7, zorder=0)

    # Collect all points for false positive / false negative computation
    # "converged" ground truth: delta_logz is within the acceptance threshold
    # For ax1: logZ threshold matches ax1_vspan (+-0.5 absolute, +-2 relative)
    #          RJD criterion: geom_mean_rjd >= 1
    # For ax2: logZ threshold matches ax2_vspan (+-0.5 absolute, +-2 relative)
    #          RJD criterion: frac_rjd_gt1 >= 0.5
    all_delta_logz = []
    all_geom_mean = []
    all_frac = []

    for i, label in enumerate(combined_data_order):
        runs = combined_data[label]
        color = default_colors[i % len(default_colors)]
        marker = 'osd'[i // len(default_colors)]
        used_colors.add(color)

        last_logz = runs[-1]['logz']

        raw_delta_logz = np.array([r['logz'] - last_logz for r in runs])
        delta_logz_err = np.array([r['logzerr'] for r in runs])

        if DELTA_LOGZ_RELATIVE_TO_ERRORBAR:
            delta_logz = np.clip(raw_delta_logz / delta_logz_err, -lnZmax, lnZmax)
        else:
            delta_logz = np.clip(raw_delta_logz, -lnZmax, lnZmax)

        geom_means = np.clip([r['geom_mean_rjd'] for r in runs], 0, 2)
        fracs = [r['frac_rjd_gt1'] for r in runs]

        all_delta_logz.extend(delta_logz.tolist())
        all_geom_mean.extend(list(geom_means))
        all_frac.extend(fracs)

        ax1.errorbar(
            delta_logz, geom_means,
            label=label,
            color=color,
            marker=marker,
            linestyle='-',
        )

        ax2.errorbar(
            delta_logz, fracs,
            label=label,
            color=color,
            marker=marker,
            linestyle='-',
        )

    ax1.set_xlim(-lnZmax, lnZmax)
    ax2.set_xlim(-lnZmax, lnZmax)
    ax1.legend(loc='upper right', numpoints=1)
    fig1.savefig('combined_geom_mean_rjd_vs_delta_logz' + ('-rel' if DELTA_LOGZ_RELATIVE_TO_ERRORBAR else '') + '.pdf')
    plt.close(fig1)

    ax2.legend(loc='upper right', numpoints=1)
    fig2.savefig('combined_frac_rjd_gt1_vs_delta_logz' + ('-rel' if DELTA_LOGZ_RELATIVE_TO_ERRORBAR else '') + '.pdf')
    plt.close(fig2)

    # --- False positive / false negative analysis ---
    # Ground truth: "converged" means |delta_logz| <= threshold
    # RJD criterion 1 (geom mean): geom_mean_rjd >= 1  -> predicts "converged"
    # RJD criterion 2 (frac):      frac_rjd_gt1 >= 0.5 -> predicts "converged"

    all_delta_logz = np.array(all_delta_logz)
    all_geom_mean = np.array(all_geom_mean)
    all_frac = np.array(all_frac)

    # Thresholds match the vertical lines drawn on the respective plots
    logz_thresh_geom = ax1_vspan_hi   # threshold for geom mean plot
    logz_thresh_frac = ax2_vspan_hi   # threshold for frac plot

    print(f"\n=== False positive / negative rates ({'relative to errorbar' if DELTA_LOGZ_RELATIVE_TO_ERRORBAR else 'absolute delta-logZ'}) ===")

    for criterion_name, logz_thresh, rjd_values, rjd_threshold, rjd_label in [
        ('geom_mean_rjd >= ', logz_thresh_geom, all_geom_mean, 1.1, 'geom mean RJD'),
        ('frac_rjd_gt1 >= ', logz_thresh_frac, all_frac, 0.666, 'frac(RJD>1)'),
    ]:
        # Ground truth: converged if |delta_logz| <= logz_thresh
        gt_converged = np.abs(all_delta_logz) <= logz_thresh
        gt_not_converged = ~gt_converged

        # Prediction: converged if rjd_value >= rjd_threshold
        pred_converged = rjd_values >= rjd_threshold
        pred_not_converged = ~pred_converged

        n_total = len(all_delta_logz)
        n_gt_converged = np.sum(gt_converged)
        n_gt_not_converged = np.sum(gt_not_converged)

        # False positive: predicted converged but actually NOT converged
        fp = np.sum(pred_converged & gt_not_converged)
        # False negative: predicted NOT converged but actually converged
        fn = np.sum(pred_not_converged & gt_converged)
        # True positive
        tp = np.sum(pred_converged & gt_converged)
        # True negative
        tn = np.sum(pred_not_converged & gt_not_converged)

        fpr = fp / n_gt_not_converged if n_gt_not_converged > 0 else float('nan')
        fnr = fn / n_gt_converged if n_gt_converged > 0 else float('nan')

        print(f"\n  Criterion: {criterion_name}{rjd_threshold}  (logZ threshold: +-{logz_thresh}, {rjd_label} threshold: {rjd_threshold})")
        print(f"    Total points: {n_total}, GT converged: {n_gt_converged}, GT not converged: {n_gt_not_converged}")
        print(f"    TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        print(f"    False positive rate (FP / GT not converged): {fpr:.3f}")
        print(f"    False negative rate (FN / GT converged):     {fnr:.3f}")


# =============================================================================
# Global Diagnostic 1: Within-sequence concordance
# =============================================================================
# Per sequence: compute Spearman rho between the RJD metric and
# |delta_logZ| / sigma (the normalised deviation from final logZ).
# Also compute within-sequence pair concordance: over all (bad, good) pairs
# in the same ladder, what fraction are correctly ordered by the RJD metric?
# "bad" = larger |delta_logZ|/sigma, "good" = smaller |delta_logZ|/sigma.
# Correctly ordered means rjd_bad > rjd_good (larger RJD for worse convergence).

print("\n" + "=" * 70)
print("GLOBAL DIAGNOSTIC 1: Within-sequence concordance")
print("=" * 70)

spearman_rhos_geom = []
spearman_rhos_frac = []
pair_concordance_geom = []
pair_concordance_frac = []

for label in combined_data_order:
    runs = combined_data[label]
    if len(runs) < 3:
        print(f"  {label}: too few runs ({len(runs)}), skipping")
        continue

    last_logz = runs[-1]['logz']
    logzerr_arr = np.array([r['logzerr'] for r in runs])
    raw_delta = np.array([r['logz'] - last_logz for r in runs])
    # Use sigma-normalised absolute deviation as the convergence error signal
    abs_dev = np.abs(raw_delta) / logzerr_arr

    geom_means = np.array([r['geom_mean_rjd'] for r in runs])
    fracs = np.array([r['frac_rjd_gt1'] for r in runs])
    run_index = np.arange(len(runs), dtype=float)

    # Spearman rho: RJD metric vs abs_dev
    # Positive rho means RJD increases with error (good: high RJD = bad convergence)
    rho_geom, _ = spearmanr(geom_means, abs_dev)
    rho_frac, _ = spearmanr(fracs, abs_dev)
    spearman_rhos_geom.append(rho_geom)
    spearman_rhos_frac.append(rho_frac)

    # Within-sequence pair concordance
    # For every pair (i, j) with i != j, if abs_dev[i] > abs_dev[j] (i is "bad"),
    # a concordant pair for geom_mean has geom_means[i] > geom_means[j].
    n = len(runs)
    concordant_geom = 0
    concordant_frac = 0
    total_pairs = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if abs_dev[i] > abs_dev[j]:
                total_pairs += 1
                if geom_means[i] < geom_means[j]:
                    concordant_geom += 1
                if fracs[i] < fracs[j]:
                    concordant_frac += 1

    pc_geom = concordant_geom / total_pairs if total_pairs > 0 else float('nan')
    pc_frac = concordant_frac / total_pairs if total_pairs > 0 else float('nan')
    pair_concordance_geom.append(pc_geom)
    pair_concordance_frac.append(pc_frac)

    print(f"\n  Sequence: {label}  (n={n} runs)")
    print(f"    Spearman rho (geom_mean_rjd vs |ΔlogZ|/σ): {rho_geom:+.3f}")
    print(f"    Spearman rho (frac_rjd_gt1  vs |ΔlogZ|/σ): {rho_frac:+.3f}")
    print(f"    Pair concordance (geom_mean_rjd): {pc_geom:.3f}  ({concordant_geom}/{total_pairs})")
    print(f"    Pair concordance (frac_rjd_gt1):  {pc_frac:.3f}  ({concordant_frac}/{total_pairs})")

if spearman_rhos_geom:
    print(f"\n  --- Distribution of Spearman rho across {len(spearman_rhos_geom)} sequences ---")
    for metric_name, rhos in [('geom_mean_rjd', spearman_rhos_geom), ('frac_rjd_gt1', spearman_rhos_frac)]:
        rhos = np.array(rhos)
        print(f"  {metric_name}: median={np.median(rhos):+.3f}, mean={np.mean(rhos):+.3f}, "
              f"min={np.min(rhos):+.3f}, max={np.max(rhos):+.3f}, "
              f"frac>0: {np.mean(rhos > 0):.2f}")
    print(f"\n  --- Distribution of pair concordance across {len(pair_concordance_geom)} sequences ---")
    for metric_name, pcs in [('geom_mean_rjd', pair_concordance_geom), ('frac_rjd_gt1', pair_concordance_frac)]:
        pcs = np.array(pcs)
        print(f"  {metric_name}: median={np.median(pcs):.3f}, mean={np.mean(pcs):.3f}, "
              f"min={np.min(pcs):.3f}, max={np.max(pcs):.3f}, "
              f"frac>0.5: {np.mean(pcs > 0.5):.2f}")

# Plot: distribution of Spearman rho across sequences
if spearman_rhos_geom:
    fig_rho, ax_rho = plt.subplots(figsize=(7, 4))
    bins_rho = np.linspace(-1, 1, 21)
    ax_rho.hist(spearman_rhos_geom, bins=bins_rho, alpha=0.6, label='geom mean RJD', color='steelblue')
    ax_rho.hist(spearman_rhos_frac, bins=bins_rho, alpha=0.6, label='frac(RJD>1)', color='darkorange')
    ax_rho.axvline(0, ls='--', color='gray', alpha=0.7)
    ax_rho.set_xlabel(r'Spearman $\rho$ (RJD metric vs $|\Delta logZ|/\sigma$, per sequence)')
    ax_rho.set_ylabel('Number of sequences')
    ax_rho.set_title(r'Within-sequence concordance: distribution of Spearman $\rho$')
    ax_rho.legend()
    fig_rho.savefig('within_sequence_spearman_rho.pdf')
    plt.close(fig_rho)

    fig_pc, ax_pc = plt.subplots(figsize=(7, 4))
    bins_pc = np.linspace(0, 1, 21)
    ax_pc.hist(pair_concordance_geom, bins=bins_pc, alpha=0.6, label='geom mean RJD', color='steelblue')
    ax_pc.hist(pair_concordance_frac, bins=bins_pc, alpha=0.6, label='frac(RJD>1)', color='darkorange')
    ax_pc.axvline(0.5, ls='--', color='gray', alpha=0.7, label='random baseline')
    ax_pc.set_xlabel('Pair concordance (fraction of bad/good pairs correctly ordered, per sequence)')
    ax_pc.set_ylabel('Number of sequences')
    ax_pc.set_title('Within-sequence pair concordance')
    ax_pc.legend()
    fig_pc.savefig('within_sequence_pair_concordance.pdf')
    plt.close(fig_pc)


# =============================================================================
# Global Diagnostic 2: Change-point alignment
# =============================================================================
# Per sequence:
#   k_err  = index of last "erratic" run  (|delta_logZ|/sigma > 2, i.e. not yet converged)
#   k_flag = index of last "flagged" run  (RJD metric indicates not converged:
#              geom_mean_rjd >= 1  OR  frac_rjd_gt1 >= 0.5)
#
# d = k_flag - k_err
#   d >= 0: flag persisted at least until the erratic phase ended (conservative, correct)
#   d <  0: flag cleared before erratic phase ended (missed erratic runs)
#
# "Overshoot" = max(d, 0): how many extra runs past stabilisation the flag persisted.
# Report: fraction with d >= 0, median d, distribution of overshoot.

print("\n" + "=" * 70)
print("GLOBAL DIAGNOSTIC 2: Change-point alignment")
print("=" * 70)

# Thresholds
ERRATIC_THRESHOLD = 2.0   # |delta_logZ|/sigma > this => erratic
RJD_GEOM_FLAGGED = 1.0    # geom_mean_rjd >= this => flagged (not converged)
RJD_FRAC_FLAGGED = 0.5    # frac_rjd_gt1 >= this => flagged (not converged)

offsets_geom = []
offsets_frac = []
overshoots_geom = []
overshoots_frac = []
missed_geom = []
missed_frac = []

for label in combined_data_order:
    runs = combined_data[label]
    if len(runs) < 3:
        continue

    n = len(runs)
    last_logz = runs[-1]['logz']
    logzerr_arr = np.array([r['logzerr'] for r in runs])
    raw_delta = np.array([r['logz'] - last_logz for r in runs])
    abs_dev = np.abs(raw_delta) / logzerr_arr

    geom_means = np.array([r['geom_mean_rjd'] for r in runs])
    fracs = np.array([r['frac_rjd_gt1'] for r in runs])

    # erratic: abs_dev > ERRATIC_THRESHOLD
    erratic = abs_dev > ERRATIC_THRESHOLD
    # flagged by each criterion
    flagged_geom = geom_means < RJD_GEOM_FLAGGED
    flagged_frac = fracs < RJD_FRAC_FLAGGED

    # k_err: last index where erratic is True; -1 if none
    erratic_indices = np.where(erratic)[0]
    k_err = int(erratic_indices[-1]) if len(erratic_indices) > 0 else -1

    # k_flag: last index where flagged is True; -1 if never flagged
    flag_indices_geom = np.where(flagged_geom)[0]
    flag_indices_frac = np.where(flagged_frac)[0]
    k_flag_geom = int(flag_indices_geom[-1]) if len(flag_indices_geom) > 0 else -1
    k_flag_frac = int(flag_indices_frac[-1]) if len(flag_indices_frac) > 0 else -1

    # Only meaningful when there was at least one erratic run
    if k_err >= 0:
        d_geom = k_flag_geom - k_err
        d_frac = k_flag_frac - k_err
        offsets_geom.append(d_geom)
        offsets_frac.append(d_frac)
        overshoots_geom.append(max(d_geom, 0))
        overshoots_frac.append(max(d_frac, 0))
        missed_geom.append(d_geom < 0)
        missed_frac.append(d_frac < 0)

        missed_n_geom = int(np.sum(erratic & ~flagged_geom))
        k_stab  = next((i for i in range(n) if not erratic[i:].any()), None)
        k_clear = next((i for i in range(n) if not flagged_geom[i:].any()), None)
        print(f"\n  Sequence: {label}  (n={n} runs)")
        print(f"    k_err={k_err}  (last erratic run index, |ΔlogZ|/σ>{ERRATIC_THRESHOLD})")
        print(f"    k_flag_geom={k_flag_geom}  d_geom={d_geom:+d}  {'MISSED' if d_geom < 0 else 'ok'}")
        print(f"    k_flag_frac={k_flag_frac}  d_frac={d_frac:+d}  {'MISSED' if d_frac < 0 else 'ok'}")
        print(f"    missed_n_geom={missed_n_geom}  k_stab={k_stab}  k_clear={k_clear}")
        # Detail: which runs were erratic?
        print(f"    Erratic run indices: {list(erratic_indices)}")
        print(f"    Flagged (geom) indices: {list(flag_indices_geom)}")
        print(f"    Flagged (frac) indices: {list(flag_indices_frac)}")
        print(r"    |ΔlogZ|/σ per run: " + "  ".join(f"{v:.2f}" for v in abs_dev))
    else:
        print(f"\n  Sequence: {label}  (n={n} runs) — no erratic runs detected, skipping change-point alignment")

if offsets_geom:
    offsets_geom = np.array(offsets_geom)
    offsets_frac = np.array(offsets_frac)
    overshoots_geom = np.array(overshoots_geom)
    overshoots_frac = np.array(overshoots_frac)
    missed_geom = np.array(missed_geom)
    missed_frac = np.array(missed_frac)

    n_seq = len(offsets_geom)
    print(f"\n  --- Change-point alignment summary ({n_seq} sequences with erratic runs) ---")
    for metric_name, offsets, overshoots, missed in [
        ('geom_mean_rjd', offsets_geom, overshoots_geom, missed_geom),
        ('frac_rjd_gt1',  offsets_frac,  overshoots_frac,  missed_frac),
    ]:
        frac_no_miss = 1.0 - np.mean(missed)
        med_offset = np.median(offsets)
        med_overshoot = np.median(overshoots)
        mean_overshoot = np.mean(overshoots)
        print(f"\n  Criterion: {metric_name}")
        print(f"    Fraction with zero missed erratic runs (d >= 0): {frac_no_miss:.3f}  ({np.sum(~missed)}/{n_seq})")
        print(f"    Median offset d = k_flag - k_err: {med_offset:+.1f}  (positive = conservative overshoot)")
        print(f"    Median overshoot (extra runs past stabilisation): {med_overshoot:.1f}")
        print(f"    Mean  overshoot (extra runs past stabilisation): {mean_overshoot:.1f}")
        print(f"    Offset distribution: min={offsets.min():+d}, median={np.median(offsets):+.1f}, max={offsets.max():+d}")

    # Plot: distribution of offsets
    fig_cp, axes_cp = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax_cp, metric_name, offsets in [
        (axes_cp[0], 'geom_mean_rjd', offsets_geom),
        (axes_cp[1], 'frac_rjd_gt1',  offsets_frac),
    ]:
        bins_cp = np.arange(offsets.min() - 0.5, offsets.max() + 1.5, 1)
        ax_cp.hist(offsets, bins=bins_cp, color='steelblue', alpha=0.8, edgecolor='white')
        ax_cp.axvline(0, ls='--', color='red', alpha=0.8, label='d=0 (exact alignment)')
        ax_cp.set_xlabel('Offset d = k_flag − k_err  (run indices)')
        ax_cp.set_ylabel('Number of sequences')
        ax_cp.set_title(f'Change-point alignment: {metric_name}')
        ax_cp.legend()
    fig_cp.suptitle('d >= 0 = conservative (flag covers erratic phase);  d < 0 = missed erratic runs', fontsize=10)
    fig_cp.tight_layout()
    fig_cp.savefig('changepoint_alignment.pdf')
    plt.close(fig_cp)
