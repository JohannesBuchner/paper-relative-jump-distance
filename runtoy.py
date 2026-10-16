import numpy as np
import argparse
from math import ceil
from ultranest.calibrator import ReactiveNestedCalibrator
from ultranest.stepsampler import SliceSampler, MHSampler
from ultranest.stepsampler import generate_random_direction, generate_cube_oriented_direction, generate_cube_oriented_differential_direction
from ultranest.stepsampler import generate_differential_direction, generate_partial_differential_direction, generate_region_oriented_direction
from ultranest.stepsampler import generate_region_random_direction, generate_mixture_random_direction, generate_region_sample_direction
from ultranest.stepsampler import OrthogonalDirectionGenerator, SequentialRegionDirectionGenerator
from ultranest.integrator import ReactiveNestedSampler
from problems import problems


def rwmsampler_with(generate_direction):
    return lambda nsteps, ndim: MHSampler(nsteps=nsteps, generate_direction=generate_direction, check_nsteps='move-distance')
def slicesampler_with(generate_direction):
    return lambda nsteps, ndim: SliceSampler(nsteps=nsteps, generate_direction=generate_direction, check_nsteps='move-distance')
def slicesampler_withclass(generate_direction_class):
    return lambda nsteps, ndim: SliceSampler(nsteps=nsteps, generate_direction=generate_direction_class(), check_nsteps='move-distance')

def ortho_slicesampler_with(generate_direction):
    return lambda nsteps, ndim: SliceSampler(nsteps=nsteps, generate_direction=OrthogonalDirectionGenerator(generate_direction), check_nsteps='move-distance')


samplers = {
    'mlfriends': None,
    'region-rwm': rwmsampler_with(generate_region_random_direction),
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


def main(args):
    paramnames, loglike, transform, vectorized, _ = problems[args.problem]
    ndim = len(paramnames)
    np.random.seed(args.seed)

    if args.nsteps is not None:
        nsteps = args.nsteps
        if isinstance(nsteps, float):
            nsteps = int(ceil(ndim * nsteps))
    else:
        nsteps = ndim

    if args.nsteps is not None:
        log_dir = 'logs/%s-%s-K%d-seed%d-nsteps%d' % (args.problem, args.stepsampler, args.num_live_points, args.seed, nsteps)
    else:
        log_dir = 'logs/%s-%s-K%d-seed%d' % (args.problem, args.stepsampler, args.num_live_points, args.seed)

    if args.stepsampler == 'mlfriends':
        sampler = ReactiveNestedSampler(
            paramnames, loglike, transform=transform, 
            log_dir=log_dir, resume=True, vectorized=vectorized)
        sampler.run(min_num_live_points=args.num_live_points, max_num_improvement_loops=0,
            viz_callback=None)
        sampler.print_results()
        if ndim <= 20:
            sampler.plot()
    elif args.nsteps is not None:
        # Run with a fixed number of nsteps, no calibration
        sampler = ReactiveNestedSampler(
            paramnames, loglike, transform=transform,
            log_dir=log_dir, resume=True, vectorized=vectorized)
        sampler.stepsampler = samplers[args.stepsampler](nsteps, ndim)
        sampler.run(min_num_live_points=args.num_live_points, max_num_improvement_loops=0,
            viz_callback=None)
        sampler.print_results()
        if sampler.stepsampler is not None:
            sampler.stepsampler.plot(filename=log_dir + '/stepsampler_stats_region.pdf')
        if ndim <= 20:
            sampler.plot()
    else:
        calibrator = ReactiveNestedCalibrator(
            paramnames, loglike, transform=transform, 
            log_dir=log_dir, resume=True, vectorized=True)
        calibrator.stepsampler = samplers[args.stepsampler](nsteps, ndim)
        
        for nstep, result in calibrator.run_iter(
            #min_calibration_loops=4,
            #max_calibration_loops=10,
            min_num_live_points=args.num_live_points, max_num_improvement_loops=0,
            viz_callback=None
        ):
            if nstep > 1500:
                break
        calibrator.sampler.print_results()
        if ndim <= 20:
            calibrator.sampler.plot()
        calibrator.plot()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--problem', choices=dict(problems).keys(), required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--num_live_points", type=int, default=400)
    parser.add_argument('--stepsampler', choices=samplers.keys(), default='mlfriends')
    parser.add_argument('--nsteps', type=float, default=None,
        help='Fix the number of steps for the step sampler. If not set, the calibrator is used. If a float, nsteps=int(ceil(ndim * nsteps)).')

    args = parser.parse_args()
    main(args)
