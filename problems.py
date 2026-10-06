import numpy as np
from numpy import pi, sin, log, cos

## speed-ups for loggamma:
import scipy.special
import functools
origgammaln = functools.lru_cache(scipy.special.gammaln)
def mygammaln(c):
    if np.shape(c) == ():
        return origgammaln(c)
    o = c.copy()
    o[:] = origgammaln(c[0])
    return o
scipy.special.gammaln = mygammaln
## end speed-up hack

import scipy.stats

def autoparamnames(ndim):
    return ['param%d' % (i+1) for i in range(ndim)]

def null_transform(x):
    return np.copy(x)


def get_asymgauss(ndim):
    sigmamin = 10**(-10 + ndim**0.5/2)
    if ndim == 4:
        assert np.isclose(sigmamin, 1e-9), sigmamin
    if ndim == 16:
        assert np.isclose(sigmamin, 1e-8), sigmamin
    if ndim == 100:
        assert np.isclose(sigmamin, 1e-5), sigmamin
    sigma = np.logspace(-1, np.log10(sigmamin), ndim)
    width = 1 - 5 * sigma
    width[width < 1e-20] = 1e-20
    centers = (np.sin(np.arange(ndim)/2.) * width + 1.) / 2.

    def loglike(theta):
        like = -0.5 * (((theta - centers)/sigma)**2).sum(axis=1) - 0.5 * np.log(2 * np.pi * sigma**2).sum()
        like[~np.isfinite(like)] = -1e300
        return like

    return autoparamnames(ndim), loglike, null_transform, True, (centers, sigma)


def get_stdfunnel(ndim, gamma=0):
    """Neal's funnel problem

    gamma controls the correlation.

    uses sigma as first parameter, and scaled deviation from mean as the others.

    """
    assert ndim > 0
    M = np.ones((ndim, ndim)) * gamma
    np.fill_diagonal(M, 1)
    Minv = np.linalg.inv(M)
    logMdet = log(np.linalg.det(M))

    prefactor = log(2 * pi) * ndim

    paramnames = ['sigma'] + ['dp%d' % i for i in range(ndim)]

    if gamma == 0:
        def loglike(theta):
            sigma = theta[0]
            x = theta[1:] * sigma
            like = -0.5 * (((x)/sigma)**2).sum() - 0.5 * log(2 * pi * sigma**2) * ndim
            return like
        def transform(x):
            z = x * 20 - 10
            z[0] = np.exp(scipy.stats.norm.ppf(x[0]) * 0.5)
            z[1:] /= z[0]
            return z
        return paramnames, loglike, transform, False, None

    def loglike_vectorized(theta):
        sigma = theta[:,0]
        var = sigma**2
        x = theta[:,1:] * sigma.reshape((-1, 1))
        r = np.einsum('ij,jk,ik->i', x, Minv, x) / var
        like = -0.5 * (prefactor + logMdet + sigma * ndim + r)
        return np.where(like > -1e300, like, -1e300)
    def transform_vectorized(x):
        z = x * 20 - 10
        z[:,0] = np.exp(scipy.stats.norm.ppf(x[:,0]) * 0.5)
        z[:,1:] /= z[:,0].reshape((-1, 1))
        return z

    return paramnames, loglike_vectorized, transform_vectorized, True, None

def get_rosenbrock(ndim):
    
    def loglike(theta):
        a = theta[:,:-1]
        b = theta[:,1:]
        return -2 * (100 * (b - a**2)**2 + (1 - a)**2).sum(axis=1)

    def transform(u):
        return u * 20 - 10
    
    return autoparamnames(ndim), loglike, transform, True, None

def get_eggbox(ndim):
    assert ndim == 2
    
    def loglike(z):
        chi = (cos(z / 2.)).prod(axis=1)
        return (2. + chi)**5

    def transform(x):
        return x * 10 * pi

    return ["a", "b"], loglike, transform, True, None

def get_beta(ndim):
    rng = np.random.RandomState(ndim)
    a_values = 10**rng.uniform(-1, 1, size=ndim)
    b_values = 10**rng.uniform(-1, 1, size=ndim)
    def loglike(theta):
        logprob = scipy.stats.beta.logpdf(theta, a=a_values, b=b_values)
        logprob[~(logprob > -1e300)] = -1e300
        logprob = logprob.sum(axis=1)
        logprob[~np.isfinite(logprob)] = -1e300
        return logprob

    return autoparamnames(ndim), loglike, null_transform, True, (a_values, b_values)

def get_symmbeta(ndim):
    rng = np.random.RandomState(ndim)
    a_values = 10**rng.uniform(-1, 1, size=ndim)
    b_values = a_values
    def loglike(theta):
        logprob = scipy.stats.beta.logpdf(theta, a=a_values, b=b_values)
        logprob[~(logprob > -1e300)] = -1e300
        logprob = logprob.sum(axis=1)
        logprob[~np.isfinite(logprob)] = -1e300
        return logprob

    return autoparamnames(ndim), loglike, null_transform, True, (a_values, b_values)

def get_loggamma(ndim):
    assert ndim >= 2
    rv1a = scipy.stats.loggamma(1, loc=2./3, scale=1./30)
    rv1b = scipy.stats.loggamma(1, loc=1./3, scale=1./30)
    rv2a = scipy.stats.norm(2./3, 1./30)
    rv2b = scipy.stats.norm(1./3, 1./30)
    rv_rest = []
    for i in range(2, ndim):
        if i <= (ndim+2)/2:
            rv = scipy.stats.loggamma(1, loc=2./3., scale=1./30)
        else:
            rv = scipy.stats.norm(2./3, 1./30)
        rv_rest.append(rv)
        del rv

    def loglike(theta):
        L1 = log(0.5 * rv1a.pdf(theta[:,0]) + 0.5 * rv1b.pdf(theta[:,0]) + 1e-300)
        L2 = log(0.5 * rv2a.pdf(theta[:,1]) + 0.5 * rv2b.pdf(theta[:,1]) + 1e-300)
        Lrest = np.sum([rv.logpdf(t) for rv, t in zip(rv_rest, theta[:,2:].transpose())], axis=0)
        like = L1 + L2 + Lrest
        like = np.where(like < -1e300, -1e300 - ((np.asarray(theta) - 0.5)**2).sum(), like)
        return like

    return autoparamnames(ndim), loglike, null_transform, True, ((rv1a, rv1b), (rv2a, rv2b), rv_rest)


def get_multisine(ncomponents, Ndata = 40, contrast = 100):
    rng = np.random.RandomState(2)
    jitter_true = 0.1
    phase_true = 0.
    period_true = 180
    amplitude_true = contrast / Ndata * jitter_true
    amplitude2_true = amplitude_true / 5
    phase2_true = 1.0
    period2_true = 44.
    offset_true = -5
    
    paramnames = {
        0 : ['offset', 'jitter'],
        1 : ['offset', 'jitter', 'amplitude', 'phase', 'period'],
        2 : ['offset', 'jitter', 'amplitude', 'phase', 'period', 'amplitude', 'phase', 'period'],
        3 : ['offset', 'jitter', 'amplitude', 'phase', 'period', 'amplitude', 'phase', 'period', 'amplitude', 'phase', 'period'],
    }[ncomponents]
    
    modelx = np.linspace(0, 360, 1000)
    modely = offset_true + amplitude_true * sin(modelx / period_true * 2 * pi + phase_true) + amplitude2_true * sin(modelx / period2_true * 2 * pi + phase2_true)
    x = rng.uniform(0, 360, Ndata)
    y = rng.normal(offset_true + amplitude_true * sin(x / period_true * 2 * pi + phase_true) + amplitude2_true * sin(x / period2_true * 2 * pi + phase2_true), jitter_true)
    data = dict(
        modelx = modelx,
        modely = modely,
        datax = x,
        datay = y,
        yerr = jitter_true,
    )
    
    def loglike(params):
        offset, jitter = params[:,:2].transpose()
        
        amplitudes, phases, periods = params[:,2::3].transpose(), params[:,3::3].transpose(), params[:,4::3].transpose()
        predicty = offset.reshape((-1, 1))
        for amplitude, phase, period in zip(amplitudes, phases, periods):
            predicty = predicty + amplitude.reshape((-1, 1)) * sin(x.reshape((1, -1)) / period.reshape((-1, 1)) * 2 * pi + phase.reshape((-1, 1)))
        
        jitter = jitter.reshape((-1, 1))
        logl = (-0.5 * log(2 * pi * jitter**2) - 0.5 * ((predicty - y) / jitter)**2).sum(axis=1)
        assert logl.shape == (len(params),), (params.shape, offset.shape, predicty.shape, x.shape, jitter.shape)
        return logl
    
    def transform(x):
        z = x.copy()
        z[:,0] = x[:,0] * 100 - 50
        z[:,1] = 10**(x[:,1] * 1 - 1.5)
        if x.shape[1] > 2:
            z[:,2::3] = 10**(x[:,2::3] * 4 - 2)
            z[:,3::3] = 2 * pi * x[:,3::3]
            z[:,4::3] = 10**(x[:,4::3] * 4 - 1)
        return z
    
    return paramnames, loglike, transform, True, data

def get_box(ndim):
    
    def loglike(theta):
        delta = np.max(theta, axis=1)
        return -0.5 * ((theta / 0.1)**2).sum(axis=1) + 100 * (delta < 0.1)

    def transform(u):
        return u
    
    return autoparamnames(ndim), loglike, transform, True, None

def get_eightschools():
    # number of schools
    J = 8
    # estimated treatment
    y = np.array([[28, 8, -3, 7, -1, 1, 18, 12]])
    # std of estimated effect
    sigma = np.array([[15, 10, 16, 11, 9, 11, 10, 18]])

    paramnames = ['tau', 'mu'] + ['theta%d' % (i+1) for i in range(J)]

    tau_prior = scipy.stats.halfcauchy(0, 5)
    mu_prior = scipy.stats.norm(0, 5)
    stdnorm = scipy.stats.norm()

    # transform unit cube (probability units) to parameters
    def transform(cube):
        params = cube.copy()
        # tau (hyper-parameter of sdv)
        #     uninformative prior tau ~ cauchy(0, 5)
        params[:, 0] = tau_prior.ppf(cube[:, 0])
        # mu (hyper-parameter of mean)
        #     mu ~ normal(0, 5)
        params[:, 1] = mu_prior.ppf(cube[:, 1])
        # thetas: treatment effect in school j
        #     thetas ~ normal(mu, tau)
        params[:, 2:] = stdnorm.ppf(cube[:, 2:]) * params[:,0].reshape((-1, 1)) + params[:,1].reshape((-1, 1))
        return params

    def loglike(params):
        # only need to compare to the data (y) here
        thetas = params[:, 2:]
        like = -0.5 * (((thetas - y)/sigma)**2).sum(axis=1)
        return like

    return paramnames, loglike, transform, True, None


problems = dict([
    ('multisine-0comp-2d', get_multisine(0)),
    ('multisine-1comp-5d', get_multisine(1)),
    ('multisine-2comp-8d', get_multisine(2)),
    ('multisine-3comp-11d', get_multisine(3)),
    ('asymgauss-4d', get_asymgauss(4)),
    ('asymgauss-16d', get_asymgauss(16)),
    ('asymgauss-100d', get_asymgauss(100)),
    ('corrfunnel-2d', get_stdfunnel(1, gamma=0.95)),
    ('corrfunnel-10d', get_stdfunnel(9, gamma=0.95)),
    ('corrfunnel-50d', get_stdfunnel(49, gamma=0.95)),
    ('rosenbrock-2d', get_rosenbrock(2)),
    ('rosenbrock-20d', get_rosenbrock(20)),
    ('rosenbrock-50d', get_rosenbrock(50)),
    ('eggbox-2d', get_eggbox(2)),
    ('beta-2d', get_beta(2)),
    ('beta-4d', get_beta(4)),
    ('beta-6d', get_beta(6)),
    ('beta-8d', get_beta(8)),
    ('beta-10d', get_beta(10)),
    ('beta-30d', get_beta(30)),
    ('symmbeta-2d', get_symmbeta(2)),
    ('symmbeta-6d', get_symmbeta(6)),
    ('symmbeta-8d', get_symmbeta(8)),
    ('symmbeta-10d', get_symmbeta(10)),
    ('symmbeta-30d', get_symmbeta(30)),
    ('loggamma-2d', get_loggamma(2)),
    ('loggamma-10d', get_loggamma(10)),
    ('loggamma-15d', get_loggamma(15)),
    ('loggamma-30d', get_loggamma(30)),
    ('box-5d', get_box(5)),
    ('eightschools-10d', get_eightschools()),
])
