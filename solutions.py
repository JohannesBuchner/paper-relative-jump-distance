import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm
from scipy.special import log_ndtr
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp
from scipy.special import erf

LOG2PI = np.log(2.0 * np.pi)

def logdiffexp(a, b):
    """log(exp(a) - exp(b)); requires a >= b."""
    if np.isscalar(a):
        if b == -np.inf:
            return a
        return a + np.log1p(-np.exp(b - a))

    out = np.empty_like(np.asarray(a, dtype=float))
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    mask = np.isneginf(b)
    out[mask] = a[mask]

    m = ~mask
    out[m] = a[m] + np.log1p(-np.exp(b[m] - a[m]))
    return out

def log_normal_interval_probability(lo, hi):
    """
    log P(lo < N(0,1) < hi), with hi >= lo.
    Stable in both left and right tails.
    """
    lo = np.asarray(lo)
    hi = np.asarray(hi)

    # If interval lies predominantly in the right tail, use
    # Phi(hi)-Phi(lo) = Phi(-lo)-Phi(-hi).
    use_sf = lo > 0

    logp_cdf = logdiffexp(log_ndtr(hi), log_ndtr(lo))
    logp_sf  = logdiffexp(log_ndtr(-lo), log_ndtr(-hi))

    return np.where(use_sf, logp_sf, logp_cdf)

def log_q(g, sigma, gamma=0.95):
    sg = np.sqrt(gamma)
    se = np.sqrt(1.0 - gamma)

    lo = (-10.0 / sigma - sg * g) / se
    hi = ( 10.0 / sigma - sg * g) / se

    return log_normal_interval_probability(lo, hi)

def corrfunnel_logz(ndim, gamma=0.95):
    """
    ndim is the argument passed to get_stdfunnel(ndim, gamma).
    Thus ndim=1 corresponds to corrfunnel-2d,
    ndim=9 corresponds to corrfunnel-10d, etc.

    Computes log Z for the likelihood exactly as written,
    including the sigma * ndim term.
    """
    sqg = np.sqrt(gamma)
    sq1mg = np.sqrt(1.0 - gamma)

    def p_box(sigma):
        # Conditional common-factor integral.
        def integrand_g(g):
            p = (
                ndtr((10.0 / sigma - sqg * g) / sq1mg)
                - ndtr((-10.0 / sigma - sqg * g) / sq1mg)
            )
            return norm.pdf(g) * p**ndim

        return quad(
            integrand_g, -10, 10,
            epsabs=1e-12, epsrel=1e-10,
            limit=200
        )[0]

    def integrand_z(z):
        sigma = np.exp(0.5 * z)

        # Normal-density measure for z:
        prior_z = norm.pdf(z)

        # This is specific to the code's sigma * ndim normalization:
        correction = np.exp(ndim * (np.log(sigma) - 0.5 * sigma))

        return prior_z * correction * p_box(sigma)

    integral = quad(
        integrand_z, -10, 10,
        epsabs=1e-12, epsrel=1e-10,
        limit=200
    )[0]

    return -ndim * np.log(20.0) + np.log(integral)

def corrfunnel_old_logz(n, gamma=0.95):
    """
    Evidence for the FIRST get_stdfunnel code:

        sigma = exp(Phi^{-1}(u0) * 0.5)
        y_i = theta_i * sigma = 20*u_i - 10
        log L = -0.5 * [
            n log(2*pi) + log|M| + n*sigma
            + y^T M^-1 y / sigma^2
        ]

    n is the function's ndim argument:
      n=1  -> corrfunnel-2d
      n=9  -> corrfunnel-10d
      n=49 -> corrfunnel-50d
    """
    sqrt_gamma = np.sqrt(gamma)
    sqrt_1mgamma = np.sqrt(1.0 - gamma)

    def p_box(sigma):
        # For n=1, avoid the unnecessary inner quadrature.
        if n == 1:
            return 2.0 * ndtr(10.0 / sigma) - 1.0

        def integrand_g(g):
            p_coordinate = (
                ndtr((10.0 / sigma - sqrt_gamma * g) / sqrt_1mgamma)
                - ndtr((-10.0 / sigma - sqrt_gamma * g) / sqrt_1mgamma)
            )
            return norm.pdf(g) * p_coordinate**n

        return quad(
            integrand_g, -10.0, 10.0,
            epsabs=1e-12, epsrel=1e-11,
            limit=300,
        )[0]

    def integrand_t(t):
        sigma = np.exp(0.5 * t)

        return (
            norm.pdf(t)
            * sigma**n
            * np.exp(-0.5 * n * sigma)
            * p_box(sigma)
        )

    integral = quad(
        integrand_t, -10.0, 10.0,
        epsabs=1e-14, epsrel=1e-11,
        limit=300,
    )[0]

    return -n * np.log(20.0) + np.log(integral)

def corrfunnelN_logz(n, gamma=0.95):
    """Evidence for the corrected Gaussian funnel likelihood."""
    sqrt_gamma = np.sqrt(gamma)
    sqrt_1mgamma = np.sqrt(1.0 - gamma)

    def p_box(sigma):
        if n == 1:
            return 2.0 * ndtr(10.0 / sigma) - 1.0

        def integrand_g(g):
            p_coordinate = (
                ndtr((10.0 / sigma - sqrt_gamma * g) / sqrt_1mgamma)
                - ndtr((-10.0 / sigma - sqrt_gamma * g) / sqrt_1mgamma)
            )
            return norm.pdf(g) * p_coordinate**n

        return quad(
            integrand_g, -10.0, 10.0,
            epsabs=1e-12, epsrel=1e-11,
            limit=300,
        )[0]

    def integrand_t(t):
        sigma = np.exp(0.5 * t)
        return norm.pdf(t) * p_box(sigma)

    integral = quad(
        integrand_t, -10.0, 10.0,
        epsabs=1e-14, epsrel=1e-11,
        limit=300,
    )[0]

    return -n * np.log(20.0) + np.log(integral)

def corrfunnel_old_logz_gh(n, gamma=0.95, ngh=120):
    """
    Evidence for the original/non-centred corrfunnel code:

        sigma = exp(0.5 * Phi^{-1}(u0))
        theta_i = (20*u_i - 10) / sigma

        logL = -0.5 * [
            n*log(2*pi) + log(det(M)) + n*sigma
            + (theta*sigma)^T M^-1 (theta*sigma) / sigma^2
        ]

    Here n is the number of correlated coordinates:
      n=1  -> corrfunnel-2d
      n=9  -> corrfunnel-10d
      n=49 -> corrfunnel-50d.
    """
    # For E[f(X)] with X~N(0,1):
    # E[f(X)] = sum_i w_i / sqrt(pi) * f(sqrt(2)*x_i).
    x, w = hermgauss(ngh)
    nodes = np.sqrt(2.0) * x
    logweights = np.log(w) - 0.5 * np.log(np.pi)

    t = nodes[:, None]      # outer log-variance-normal coordinate
    g = nodes[None, :]      # common equicorrelation factor

    sigma = np.exp(0.5 * t)

    sqrt_gamma = np.sqrt(gamma)
    sqrt_1mgamma = np.sqrt(1.0 - gamma)

    lower = (-10.0 / sigma - sqrt_gamma * g) / sqrt_1mgamma
    upper = ( 10.0 / sigma - sqrt_gamma * g) / sqrt_1mgamma

    logq = log_normal_interval_probability(lower, upper)

    # log of:
    # sigma**n * exp(-n*sigma/2) * q(g, sigma)**n
    logintegrand = (
        n * np.log(sigma)
        - 0.5 * n * sigma
        + n * logq
    )

    logjointweights = logweights[:, None] + logweights[None, :]

    log_integral = logsumexp(logjointweights + logintegrand)

    return -n * np.log(20.0) + log_integral


def box_logz(d):
    return 100 + d * np.log(0.1 * np.sqrt(np.pi / 2) * erf(1 / np.sqrt(2)))


if __name__ == '__main__':
    for n, label in [(1, "corrfunnel-2d"),
                     (9, "corrfunnel-10d"),
                     (49, "corrfunnel-50d")]:
        #logz = corrfunnel_old_logz(n)
        logz = corrfunnelN_logz(n)
        print(label, "logZ =", logz, " Z =", np.exp(logz))
        #for ngh in [40, 60, 80, 120, 160, 200]:
        #    print(ngh, corrfunnel_old_logz_gh(n, gamma=0.95, ngh=ngh))

    #for d in 2, 10, 50:
    #    print(d, corrfunnel_logz(d))
