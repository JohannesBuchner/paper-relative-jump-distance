"""Calculates the Bayesian evidence and posterior samples of arbitrary monomodal models."""

from __future__ import print_function
from __future__ import division

import time
import os
import numpy as np
import corner
import matplotlib.pyplot as plt
import json

from ultranest.calibrator import ReactiveNestedCalibrator
from ultranest.stepsampler import SliceSampler
from ultranest.stepsampler import generate_random_direction, generate_cube_oriented_direction, generate_cube_oriented_differential_direction
from ultranest.stepsampler import generate_differential_direction, generate_partial_differential_direction, generate_region_oriented_direction
from ultranest.stepsampler import generate_region_random_direction, generate_mixture_random_direction, generate_region_sample_direction
from ultranest.stepsampler import OrthogonalDirectionGenerator, SequentialRegionDirectionGenerator
from ultranest.integrator import ReactiveNestedSampler


def slicesampler_with(generate_direction):
    return lambda nsteps, ndim: SliceSampler(nsteps=nsteps, generate_direction=generate_direction, check_nsteps='move-distance')
def slicesampler_withclass(generate_direction_class):
    return lambda nsteps, ndim: SliceSampler(nsteps=nsteps, generate_direction=generate_direction_class(), check_nsteps='move-distance')

def ortho_slicesampler_with(generate_direction):
    return lambda nsteps, ndim: SliceSampler(nsteps=nsteps, generate_direction=OrthogonalDirectionGenerator(generate_direction), check_nsteps='move-distance')


samplers = {
    'mlfriends': None,
    'cube-slice': slicesampler_with(generate_cube_oriented_direction),
    'region-slice': slicesampler_with(generate_region_oriented_direction),
    'region-seq-slice': slicesampler_withclass(SequentialRegionDirectionGenerator),
    'cube-harm': slicesampler_with(generate_random_direction),
    'region-harm': slicesampler_with(generate_region_random_direction),
    'cube-ortho-harm': ortho_slicesampler_with(generate_random_direction),
    'region-ortho-harm': ortho_slicesampler_with(generate_region_random_direction),
    'region-prop-ortho-slice': ortho_slicesampler_with(generate_region_sample_direction),
    'de-harm': slicesampler_with(generate_differential_direction),
    'de1': slicesampler_with(generate_cube_oriented_differential_direction),
    'depart-harm': slicesampler_with(generate_partial_differential_direction),
    'de-mix': slicesampler_with(generate_mixture_random_direction),
}

class RegionLogger():
    def __init__(self, prefix):
        self.prefix = prefix
    
    def __call__(self, points, info, region, transformLayer, region_fresh=False):
        filename = self.prefix + 'region_%d.txt.gz' % info['it']
        np.savetxt(filename, region.u)


def ultranest_run(sampler, log_dir, max_ncalls, **kwargs):
    offset = 0
    # first quick run-through:
    for i, results_ in enumerate(sampler.run_iter(
        frac_remain=0.5, max_num_improvement_loops=0, 
        max_ncalls=max_ncalls, viz_callback=RegionLogger(log_dir), **kwargs)
    ):
        results = dict(results_)
        samples = results.pop('samples')
        del results['weighted_samples']
        param_names = results['paramnames']
        with open(log_dir + '/results-%d.json' % i, 'w') as fout:
            json.dump(results, fout, indent=4)
        np.savetxt(log_dir + '/samples-%d.txt.gz' % i, samples, delimiter=',', header=','.join(param_names))
        offset = i + 1

    for i, results_ in enumerate(sampler.run_iter(max_ncalls=max_ncalls, viz_callback=RegionLogger(log_dir), **kwargs), start=offset):
        # complete run-through, with potential reactive iterations
        results = dict(results_)
        samples = results.pop('samples')
        del results['weighted_samples']
        param_names = results['paramnames']
        with open(log_dir + '/results-%d.json' % i, 'w') as fout:
            json.dump(results, fout, indent=4)
        np.savetxt(log_dir + '/samples-%d.txt.gz' % i, samples, delimiter=',', header=','.join(param_names))

    with open(log_dir + '/results.json', 'w') as fout:
        json.dump(results, fout, indent=4)
    np.savetxt(log_dir + '/samples.txt.gz', samples, delimiter=',', header=','.join(param_names))

    return results_


def run_sampler(
    param_names,
    loglike,
    transform=None,
    vectorized=False,
    max_ncalls=400000000
):
    """Initialise sampler.

    Parameters
    -----------
    param_names: list of str, names of the parameters.
        Length gives dimensionality of the sampling problem.

    loglike: function
        log-likelihood function.
        Receives multiple parameter vectors, returns vector of likelihood.
    transform: function
        parameter transform from unit cube to physical parameters.
        Receives multiple cube vectors, returns multiple parameter vectors.
    vectorized: bool
        if true, likelihood and transform receive arrays of points, and return arrays
    max_ncalls: int
        maximum number of likelihood evaluations
    """
    samplername = os.environ['SAMPLER']
    log_dir = os.environ['LOGDIR']
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    ndim = len(param_names)
    
    if vectorized:
        def flat_loglike(theta):
            L = loglike(theta.reshape((1, -1)))[0]
            assert np.isfinite(L), L
            return L
        
        def flat_transform(cube):
            return transform(cube.reshape((1, -1)))[0]
    else:
        flat_transform = transform
        flat_loglike = loglike

    if samplername == 'testsampler':
        for i in range(2):
            print("test sampler: drawing one prior sample")
            tstart = time.time()
            u = np.random.uniform(size=ndim)
            print("  transforming unit cube sample:", u)
            p = flat_transform(u)
            print("  likelihood evaluation of parameters:", p)
            L = flat_loglike(p)
            print("  log-likelihood is:", L)
            print("  --> computational cost: %.5f seconds" % (time.time() - tstart))
    elif samplername == 'ultranest-safe':
        from ultranest import ReactiveNestedSampler
        sampler = ReactiveNestedSampler(param_names, loglike, transform=transform, 
            log_dir=log_dir, resume=True, vectorized=vectorized, ndraw_max=10000000)
        ultranest_run(sampler, log_dir, max_ncalls)
        sampler.print_results()
        if ndim <= 30:
            sampler.plot()
    elif samplername == 'ultranest-fast-fixed100':
        from ultranest import ReactiveNestedSampler
        slice_steps = 100
        sampler = ReactiveNestedSampler(param_names, loglike, transform=transform,
            log_dir=log_dir, resume=True, vectorized=vectorized)
        import ultranest.stepsampler
        sampler.stepsampler = ultranest.stepsampler.SliceSampler(
            nsteps=slice_steps, region_filter=False,
            generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
            log=open(log_dir + '/stepsampler.log', 'w')
        )
        
        ultranest_run(sampler, log_dir, max_ncalls)
        sampler.print_results()
        if sampler.stepsampler is not None:
            sampler.stepsampler.plot(filename = log_dir + '/stepsampler_stats_region.pdf')
        if ndim <= 30:
            sampler.plot()
    elif samplername == 'ultranest-calibrator':
        calibrator = ReactiveNestedCalibrator(
            param_names, loglike, transform=transform, 
            log_dir=log_dir, resume=True, vectorized=vectorized)
        calibrator.stepsampler = samplers[os.environ['STEPSAMPLER']](ndim, ndim)
        
        for nstep, result in calibrator.run(max_num_improvement_loops=0, viz_callback=None):
            if nstep > 1500:
                break
        calibrator.sampler.print_results()
        if ndim <= 20:
            calibrator.sampler.plot()
        calibrator.plot()
    elif samplername in ('goodman-weare', 'slice'):
        from autoemcee import ReactiveAffineInvariantSampler
        sampler = ReactiveAffineInvariantSampler(
            param_names, loglike, transform=transform, 
            sampler=samplername, vectorized=vectorized)
        results = dict(sampler.run(max_ncalls=max_ncalls))
        sampler.print_results()
        samples = results.pop('samples')
        json.dump(results, open(log_dir + '/results.json', 'w'), indent=4)
        np.savetxt(log_dir + '/samples.txt.gz', samples, delimiter=',', header=','.join(param_names))
        if ndim <= 30:
            sampler.plot()
            plt.savefig(log_dir + '/corner.pdf', bbox_inches='tight')
            plt.close()
    elif samplername == 'vbis' or samplername == 'vbis-wide':
        from snowline import ReactiveImportanceSampler
        sampler = ReactiveImportanceSampler(param_names, flat_loglike, transform=flat_transform)
        i = 0
        results = dict(sampler.laplace_approximate(
            num_global_samples=ndim * 1000 * (10 if samplername == 'vbis-wide' else 1)))
        samples = results.pop('samples')
        with open(log_dir + '/results-%d.json' % i, 'w') as fout:
            json.dump(results, fout, indent=4)
        np.savetxt(log_dir + '/samples-%d.txt.gz' % i, samples, delimiter=',', header=','.join(param_names))

        for i, results_ in enumerate(sampler.run_iter(
            num_gauss_samples=ndim * 200,
            max_ncalls=max_ncalls,
            min_ess=400,
            max_improvement_loops=40,
            heavytail_laplaceapprox=(samplername == 'vbis-wide'),
        ), start=1):
            results = dict(results_)
            samples = results.pop('samples')
            with open(log_dir + '/results-%d.json' % i, 'w') as fout:
                json.dump(results, fout, indent=4)
            np.savetxt(log_dir + '/samples-%d.txt.gz' % i, samples, delimiter=',', header=','.join(param_names))

        sampler.print_results()
        json.dump(results, open(log_dir + '/results.json', 'w'), indent=4)
        np.savetxt(log_dir + '/samples.txt.gz', samples, delimiter=',', header=','.join(param_names))
        if ndim <= 30:
            corner.corner(
                samples,
                labels=param_names,
                show_titles=True)
            plt.savefig(log_dir + '/corner.pdf', bbox_inches='tight')
            plt.close()
    elif samplername == 'vegas':
        import vegas
        integ = vegas.Integrator([[0, 1]] * len(param_names))
        result = integ(lambda u: np.exp(flat_loglike(flat_transform(u))), nitn=10, neval=100000)
        print(result.summary())
        print('result = %s    Q = %.2f' % (result, result.Q))
        print('lnZ:', np.log(result.mean))
    elif samplername == 'lhsgrid':
        import scipy.stats
        sampler = scipy.stats.qmc.LatinHypercube(len(param_names))
        usamples = sampler.random(n=40000000)
        if vectorized:
            psamples = transform(usamples)
            L = loglike(psamples)
            lnZ = np.log(np.exp(L - L.max()).mean()) + L.max()
        info = dict(
            ncall=len(usamples), 
            maximum_likelihood=dict(L=float(L.max()), u=usamples[np.argmax(L)].tolist(), p=psamples[np.argmax(L)].tolist()),
            logz=float(lnZ))
        print(lnZ)
        json.dump(info, open(log_dir + '/results.json', 'w'), indent=4)
    else:
        assert False, ("unknown sampler:", samplername)

        
if __name__ == '__main__':
    
    param_names = ['Hinz', 'Kunz']

    def loglike(z):
        a = -0.5 * (((z - 0.5) / 0.01)**2).sum() + -0.5 * ((z[0] - z[1])/0.01)**2
        return a

    def transform(x):
        return 10. * x - 5.
    
    run_sampler(param_names, loglike, transform)
