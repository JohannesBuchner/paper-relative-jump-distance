import numpy as np
#from numpy import log, exp, log10
#from ultranest.mlfriends import MLFriends, AffineLayer
import joblib
import matplotlib.pyplot as plt

mem = joblib.Memory('.', verbose=False)

def sample_ellipsoid(nsamples, ndim, sigma=0.01, center=0.5):
	""" Sample from a unit sphere with constant probability density """
	z = np.random.normal(size=(nsamples, ndim))
	z /= ((z**2).sum(axis=1)**0.5).reshape((nsamples, 1))
	u = z * np.random.uniform(size=(nsamples, 1))**(1./ndim)
	return u * sigma + center


@mem.cache
def evaluate_medianneighbourdist(d, sigma, Nsamples=40):
	u = sample_ellipsoid(Nsamples, d, sigma=sigma)
	a = u[::2]
	b = u[1::2]
	return np.median(np.linalg.norm(a - b))

@mem.cache
def evaluate_meanneighbourdist(d, sigma, Nsamples=40):
	u = sample_ellipsoid(Nsamples, d, sigma=sigma)
	a = u[::2]
	b = u[1::2]
	return np.mean(np.linalg.norm(a - b)**2)**0.5

def evaluate_all(ds):
	sigma = 0.01

	sequence = [evaluate_medianneighbourdist(d, sigma, Nsamples=2000) for d in ds]
	plt.plot(ds, sequence)
	plt.ylim(0, None)
	plt.ylabel('Median distance between live points')
	plt.xlabel('Dimensionality')
	plt.xscale('log')
	#plt.yscale('log')
	plt.savefig('ellipsoidn.pdf')
	plt.close()

if __name__ == '__main__':
	evaluate_all(ds = [2, 4, 8, 10, 16, 20, 32, 50, 64, 128])
