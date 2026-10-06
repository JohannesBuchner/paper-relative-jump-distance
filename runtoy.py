import numpy as np
import argparse
from ultranest.calibrator import ReactiveNestedCalibrator
from ultranest.stepsampler import SliceSampler
from ultranest.stepsampler import generate_random_direction, generate_cube_oriented_direction, generate_cube_oriented_differential_direction
from ultranest.stepsampler import generate_differential_direction, generate_partial_differential_direction, generate_region_oriented_direction
from ultranest.stepsampler import generate_region_random_direction, generate_mixture_random_direction, generate_region_sample_direction
from ultranest.stepsampler import OrthogonalDirectionGenerator, SequentialRegionDirectionGenerator
from ultranest.integrator import ReactiveNestedSampler
from problems import problems


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


def main(args):
    paramnames, loglike, transform, vectorized, _ = problems[args.problem]
    ndim = len(paramnames)
    np.random.seed(args.seed)

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
    else:
        calibrator = ReactiveNestedCalibrator(
            paramnames, loglike, transform=transform, 
            log_dir=log_dir, resume=True, vectorized=True)
        calibrator.stepsampler = samplers[args.stepsampler](ndim, ndim)
        
        for nstep, result in calibrator.run_iter(min_num_live_points=args.num_live_points, max_num_improvement_loops=0,
            viz_callback=None):
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

    args = parser.parse_args()
    main(args)
