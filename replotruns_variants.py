import json
import re
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from solutions import corrfunnelN_logz

# extract prefix from systematiclogs/ligo/ultranest-calibrator/-nsteps7/plots/stepsampler.pdf.txt.gz
# can become something like logs/rosenbrock-20d-cube-slice-K400-seed42-nsteps20
prefixes = {
    filename.replace('/plots/stepsampler.pdf.txt.gz', '')
        for filename in sys.argv[1:]}

# Detect variant mode: all prefixes share the same base problem name
# A variant differs by dimension, seed, or number of live points (K)
def get_base_problem(output_dir):
    """Extract base problem name, stripping variant suffixes like seed, K, dimension."""
    label = output_dir.rstrip('/').split('/')[-1]
    # Remove seed suffix
    label = re.sub(r'-seed\d+', '', label)
    # Remove K (live points) suffix
    label = re.sub(r'-K\d+', '', label)
    # Remove dimension suffix
    label = re.sub(r'-\d+d', '', label)
    return label

def get_variant_label(output_dir, all_dirs):
    """Extract what changed in this variant, omitting parts that are the same across all variants."""
    label = output_dir.rstrip('/').split('/')[-1]

    # Collect values for each part across all dirs
    all_dims = set()
    all_Ks = set()
    all_seeds = set()
    for d in all_dirs:
        l = d.rstrip('/').split('/')[-1]
        m = re.search(r'-(\d+)d', l)
        if m:
            all_dims.add(m.group(1))
        m = re.search(r'-K(\d+)', l)
        if m:
            all_Ks.add(m.group(1))
        m = re.search(r'-seed(\d+)', l)
        if m:
            all_seeds.add(m.group(1))

    parts = []
    m = re.search(r'-(\d+)d', label)
    if m and len(all_dims) > 1:
        parts.append('d=%s' % m.group(1))
    m = re.search(r'-K(\d+)', label)
    if m and len(all_Ks) > 1:
        parts.append('K=%s' % m.group(1))
    m = re.search(r'-seed(\d+)', label)
    if m and len(all_seeds) > 1:
        parts.append('seed=%s' % m.group(1))
    return ', '.join(parts) if parts else label

def get_varied_suffix(variant_dirs):
    """Return a suffix string indicating which parameters varied, e.g. _seed_K_d."""
    all_dims = set()
    all_Ks = set()
    all_seeds = set()
    for d in variant_dirs:
        l = d.rstrip('/').split('/')[-1]
        m = re.search(r'-(\d+)d', l)
        if m:
            all_dims.add(m.group(1))
        m = re.search(r'-K(\d+)', l)
        if m:
            all_Ks.add(m.group(1))
        m = re.search(r'-seed(\d+)', l)
        if m:
            all_seeds.add(m.group(1))

    parts = []
    if len(all_seeds) > 1:
        parts.append('seed')
    if len(all_Ks) > 1:
        parts.append('K')
    if len(all_dims) > 1:
        parts.append('d')
    if parts:
        return '_' + '_'.join(parts)
    return ''

def is_d_varied(variant_dirs):
    """Return True if dimension varies across the variant dirs."""
    all_dims = set()
    for d in variant_dirs:
        l = d.rstrip('/').split('/')[-1]
        m = re.search(r'-(\d+)d', l)
        if m:
            all_dims.add(m.group(1))
    return len(all_dims) > 1

# Group prefixes by base problem name (stripping nsteps)
output_dirs = {}
for filename in sys.argv[1:]:
    prefix = filename.replace('/plots/stepsampler.pdf.txt.gz', '')
    output_dir = prefix.split('-nsteps', 1)[0]
    base = get_base_problem(output_dir)
    if base not in output_dirs:
        output_dirs[base] = []
    if output_dir not in output_dirs[base]:
        output_dirs[base].append(output_dir)

print(output_dirs)
completed = set()

for base_problem, variant_dirs in output_dirs.items():
    # All variants go into one plot
    # First pass: collect all nsteps values across all variants
    all_nsteps = set()
    for output_dir in variant_dirs:
        example_run_prefix = None
        for filename in sys.argv[1:]:
            p = filename.replace('/plots/stepsampler.pdf.txt.gz', '')
            if p.split('-nsteps', 1)[0] == output_dir:
                example_run_prefix = p
                break
        if example_run_prefix is None:
            continue
        if not os.path.exists(example_run_prefix + '/info/results.json'):
            continue
        info = json.load(open(example_run_prefix + '/info/results.json'))
        nparams = len(info['paramnames'])
        for K in range(-1, 100):
            nsteps = int(2**K * nparams)
            run_prefix = output_dir + ('-nsteps%d/' % nsteps)
            if not os.path.exists(run_prefix + '/info/results.json'):
                if K == -1:
                    continue
                break
            all_nsteps.add(nsteps)

    all_nsteps_sorted = sorted(all_nsteps)
    nsteps_color_map = {}
    cmap = plt.get_cmap('tab10')
    for i, ns in enumerate(all_nsteps_sorted):
        nsteps_color_map[ns] = cmap(i % 10)

    n_variants = len(variant_dirs)
    fig, ax = plt.subplots(figsize=(7, 1.5 + 1.4 * n_variants))

    global_bins = None
    global_binlo = None
    global_binhi = None

    # First pass to determine bins
    for output_dir in variant_dirs:
        example_run_prefix = None
        for filename in sys.argv[1:]:
            p = filename.replace('/plots/stepsampler.pdf.txt.gz', '')
            if p.split('-nsteps', 1)[0] == output_dir:
                example_run_prefix = p
                break
        if example_run_prefix is None:
            continue
        if not os.path.exists(example_run_prefix + '/info/results.json'):
            continue
        info = json.load(open(example_run_prefix + '/info/results.json'))
        nparams = len(info['paramnames'])
        for K in range(0, 100):
            nsteps = 2**K * nparams
            run_prefix = output_dir + ('-nsteps%d/' % nsteps)
            if not os.path.exists(run_prefix + '/info/results.json'):
                break
            filename_csv = run_prefix + '/plots/stepsampler.pdf.txt.gz'
            df = pd.read_csv(filename_csv)
            distance = df['jump-distance']
            reference_distance = df['reference-distance']
            rjd = distance / reference_distance
            geom_mean_rjd = np.exp(np.mean(np.log(rjd + 1e-3)))
            if global_bins is None:
                binlo = max(0.0001, min(0.1, geom_mean_rjd / 10, np.quantile(rjd, 0.02)))
                binhi = min(100, max(4, geom_mean_rjd * 4, np.quantile(rjd, 0.98)))
                global_binlo = binlo
                global_binhi = binhi
                global_bins = np.geomspace(binlo, binhi, 60)
        if global_bins is not None:
            break

    bins = global_bins
    binlo = global_binlo
    binhi = global_binhi

    offset_step = 1.0
    legend_handles = {}

    # Collect per-variant run data for insets: {output_dir: [(nsteps, logz, logzerr, color), ...]}
    variant_run_data = {od: [] for od in variant_dirs}

    for v_idx, output_dir in enumerate(variant_dirs):
        variant_label = get_variant_label(output_dir, variant_dirs)
        offset = v_idx * offset_step

        example_run_prefix = None
        for filename in sys.argv[1:]:
            p = filename.replace('/plots/stepsampler.pdf.txt.gz', '')
            if p.split('-nsteps', 1)[0] == output_dir:
                example_run_prefix = p
                break
        if example_run_prefix is None:
            continue
        if not os.path.exists(example_run_prefix + '/info/results.json'):
            continue
        info = json.load(open(example_run_prefix + '/info/results.json'))
        nparams = len(info['paramnames'])

        last_N = None
        last_color = None

        for K in range(-1, 100):
            nsteps = int(2**K * nparams)
            run_prefix = output_dir + ('-nsteps%d/' % nsteps)
            if not os.path.exists(run_prefix + '/info/results.json'):
                if K == -1:
                    continue
                break
            result = json.load(open(run_prefix + '/info/results.json'))
            filename_csv = run_prefix + '/plots/stepsampler.pdf.txt.gz'

            df = pd.read_csv(filename_csv)

            distance = df['jump-distance']
            reference_distance = df['reference-distance']
            rjd = distance / reference_distance
            geom_mean_rjd = np.exp(np.mean(np.log(rjd + 1e-3)))
            frac_rjd_gt1 = np.mean(rjd > 1)

            color = nsteps_color_map[nsteps]

            counts, _ = np.histogram(np.clip(rjd, binlo, binhi), bins=bins)
            max_count = counts.max() if counts.max() > 0 else 1
            norm_counts = counts / max_count * 0.85 + offset

            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            lw = 2 if K < 7 else 1
            ls = '-' if K < 7 else '--'
            line, = ax.step(bins[:-1], norm_counts, where='post', color=color, lw=lw, ls=ls)
            if nsteps not in legend_handles:
                legend_handles[nsteps] = (line, str(nsteps))

            idx = np.searchsorted(bins, geom_mean_rjd) - 1
            idx = np.clip(idx, 0, len(norm_counts) - 1)
            # ax.plot(geom_mean_rjd, norm_counts[idx] * 1.04, 'v', color=color, ms=8, mfc='white')
            ax.plot(geom_mean_rjd, 0.90 + offset, 'v', color=color, ms=8, mfc='white')

            variant_run_data[output_dir].append((nsteps, result['logz'], result['logzerr'], color))

            last_N = norm_counts
            last_color = color

        label_position = 'left'
        if last_N is not None:
            if label_position == 'left':
                right_val = last_N[0]
                ax.text(bins[0] * 1.04, right_val + 0.04, variant_label,
                        color='black', fontsize=14, va='bottom', ha='left', weight="bold", family="sans-serif",
                        )
            else:
                right_val = last_N[-1]
                ax.text(bins[-1] / 1.04, right_val - 0.04, variant_label,
                        color='black', fontsize=14, va='bottom', ha='right')

    ax.set_xscale('log')
    ax.set_xlabel('Relative jump distance (RJD)')
    ax.set_xlim(bins[0], bins[-1])

    # Integer yticks: one tick per variant offset level
    ytick_positions = [v_idx * offset_step for v_idx in range(n_variants + 1)]
    ytick_labels = [str(v_idx) for v_idx in range(n_variants + 1)]
    ax.set_yticks(ytick_positions)
    ax.set_ylim(-0.04, n_variants + 0.04)
    ax.set_yticklabels(ytick_labels)
    ax.set_ylabel('Normalised frequency + offset')

    sorted_nsteps = sorted(legend_handles.keys())
    handles = [legend_handles[ns][0] for ns in sorted_nsteps]
    labels = [legend_handles[ns][1] for ns in sorted_nsteps]
    ax.legend(handles, labels, loc='upper right', title='steps', fontsize=7)

    # Add one inset axis per variant for ln(Z) convergence
    # Place insets along the left side of the main axes, one per variant row
    # Inset positions in figure-fraction coordinates; we use ax.get_position() after drawing
    fig.canvas.draw()
    ax_pos = ax.get_position()  # Bbox in figure fraction

    inset_w = 0.13
    inset_h = 0.08
    inset_x_left = ax_pos.x0 + 0.2  # just inside left edge of axes

    # Determine true logZ reference line per variant (same logic as replotruns.py)
    def get_true_logz(output_dir):
        if 'beta' in output_dir or 'asymgauss' in output_dir or 'loggamma' in output_dir:
            return 0.0
        if 'eggbox' in output_dir:
            return 235.88
        label_part = output_dir.rstrip('/').split('/')[-1]
        m = re.search(r'corrfunnelN-(\d+)d', label_part)
        if m:
            total_dim = int(m.group(1))
            ndim_arg = total_dim - 1
            if ndim_arg < 50:
                return corrfunnelN_logz(ndim_arg)
        return None

    # Determine whether d is varied across variant_dirs
    d_varied = is_d_varied(variant_dirs)

    # First pass: collect ylim for each inset so we can unify them (unless d is varied)
    inset_ylims = []
    for v_idx, output_dir in enumerate(variant_dirs):
        runs = variant_run_data[output_dir]
        if not runs:
            inset_ylims.append(None)
            continue

        logzs = [r[1] for r in runs]
        logzerrs = [r[2] for r in runs]

        # Compute natural ylim from data
        all_lo = [lz - lze for lz, lze in zip(logzs, logzerrs)]
        all_hi = [lz + lze for lz, lze in zip(logzs, logzerrs)]
        true_logz = get_true_logz(output_dir)
        if true_logz is not None:
            all_lo.append(true_logz)
            all_hi.append(true_logz)

        ylo_i = min(all_lo)
        yhi_i = max(all_hi)
        margin = (yhi_i - ylo_i) * 0.1 if yhi_i > ylo_i else 0.1
        ylo_i -= margin
        yhi_i += margin

        # Expand if too narrow
        if yhi_i - ylo_i < 1.2:
            mid = (ylo_i + yhi_i) / 2
            ylo_i = mid - 0.6
            yhi_i = mid + 0.6

        inset_ylims.append((ylo_i, yhi_i))

    # Compute unified ylim across all variants only if d is not varied
    if not d_varied:
        valid_ylims = [yl for yl in inset_ylims if yl is not None]
        if valid_ylims:
            unified_ylo = min(yl[0] for yl in valid_ylims)
            unified_yhi = max(yl[1] for yl in valid_ylims)
            # Ensure minimum span
            if unified_yhi - unified_ylo < 1.2:
                mid = (unified_ylo + unified_yhi) / 2
                unified_ylo = mid - 0.6
                unified_yhi = mid + 0.6
        else:
            unified_ylo, unified_yhi = -0.6, 0.6
        use_unified = True
    else:
        use_unified = False

    # Retrieve the final axes y-limits once, for coordinate conversion
    ylo_data, yhi_data = ax.get_ylim()

    inset_shift = 0.2  # shift insets up by this many data coordinates

    for v_idx, output_dir in enumerate(variant_dirs):
        runs = variant_run_data[output_dir]
        if not runs:
            continue

        variant_label = get_variant_label(output_dir, variant_dirs)

        # The variant strip occupies data y range [offset .. offset + 1], i.e. [v_idx .. v_idx+1].
        # Place the inset centre at a fractional position within that strip, shifted up by inset_shift.
        inset_rel_y = 0.5  # fractional position within [offset .. offset+1]; 0=bottom, 1=top
        strip_bottom_data = v_idx * offset_step
        data_y_centre = strip_bottom_data + inset_rel_y * offset_step + inset_shift

        # Convert data coords to axes fraction, then to figure fraction
        axes_frac_y = (data_y_centre - ylo_data) / (yhi_data - ylo_data)
        fig_frac_y = ax_pos.y0 + axes_frac_y * ax_pos.height

        inset_bottom = fig_frac_y - inset_h / 2

        # Clamp so the inset stays within the figure axes area
        inset_bottom = max(ax_pos.y0, min(ax_pos.y0 + ax_pos.height - inset_h, inset_bottom))

        ax_inset = fig.add_axes([inset_x_left, inset_bottom, inset_w, inset_h])
        ax_inset.set_facecolor('#F5F5F5')
        ax_inset.tick_params(labelsize=5)
        ax_inset.set_xlabel('steps', fontsize=5)

        nsteps_list = [r[0] for r in runs]
        logzs = [r[1] for r in runs]
        logzerrs = [r[2] for r in runs]
        colors = [r[3] for r in runs]

        for i, (ns, lz, lze, col) in enumerate(runs):
            ax_inset.errorbar(x=ns, y=lz, yerr=lze, color=col, marker='x', ms=3, lw=0.8, capsize=2)
            if i >= 1:
                ax_inset.plot([nsteps_list[i-1], ns], [logzs[i-1], lz], color='k', lw=0.5)

        true_logz = get_true_logz(output_dir)
        if true_logz is not None:
            ax_inset.axhline(true_logz, ls=':', color='gray', alpha=0.5, lw=0.8)

        # Apply ylim: unified if d is not varied, per-variant otherwise
        if use_unified:
            ylo_use, yhi_use = unified_ylo, unified_yhi
        else:
            yl = inset_ylims[v_idx]
            if yl is not None:
                ylo_use, yhi_use = yl
            else:
                ylo_use, yhi_use = -0.6, 0.6

        ax_inset.set_ylim(ylo_use, yhi_use)

        # Integer yticks for inset
        int_ticks = [t for t in range(int(np.floor(ylo_use)), int(np.ceil(yhi_use)) + 1)]
        if len(int_ticks) > 5:
            step = max(1, int(np.ceil((yhi_use - ylo_use) / 4)))
            int_ticks = list(range(int(np.floor(ylo_use)), int(np.ceil(yhi_use)) + 1, step))
        ax_inset.set_yticks(int_ticks)
        ax_inset.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: '%d' % v))

        ax_inset.set_ylabel(r'$\ln Z$')
    problem_name = base_problem
    for od in variant_dirs:
        label_part = od.rstrip('/').split('/')[-1]
        m = re.match(r'([a-zA-Z]+)', label_part)
        if m:
            problem_name = m.group(1)
            break

    varied_suffix = get_varied_suffix(variant_dirs)
    output_filename = '%s_variants%s.pdf' % (problem_name, varied_suffix)
    print("writing", output_filename)
    plt.savefig(output_filename)
    plt.close()
