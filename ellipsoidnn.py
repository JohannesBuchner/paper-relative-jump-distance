import numpy as np
#from numpy import log, exp, log10
from ultranest.mlfriends import MLFriends, AffineLayer
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
def evaluate_mlfriendswrapper(d, Nlive, sigma, Nsamples=40):
	layer = AffineLayer()
	radii = []
	for trials in range(Nsamples):
		u = sample_ellipsoid(Nlive, d, sigma=sigma)
		try:
			region = MLFriends(u, layer)
			r, f = region.compute_enlargement()  
			layer2 = layer.create_new(u, r)
			region2 = MLFriends(u, layer2)
			region2.maxradiussq, region2.enlarge = region2.compute_enlargement()  
			region2.create_ellipsoid()
			radius = np.sqrt(region2.maxradiussq)
		except AssertionError:
			radius = np.nan
		except np.linalg.LinAlgError:
			radius = np.nan
		radii.append(radius)
	return np.array(radii)

def evaluate_all(ds, Nlives):
	sigma = 0.01

	outputs = []
	for d in ds:
		sequence = []
		sequence_errs = []
		for Nlive in Nlives:
			if Nlive < d:
				radii = [np.nan]*3
			else:
				radii = evaluate_mlfriendswrapper(d, Nlive, sigma=sigma)
			print(d, Nlive, np.nanmean(radii), np.nanstd(radii))
			outputs.append([d, Nlive, np.nanmean(radii), np.nanstd(radii)])
			sequence.append(np.nanmean(radii))
			sequence_errs.append(np.nanstd(radii))
		sequence = np.array(sequence)
		sequence_errs = np.array(sequence_errs)
		p = plt.errorbar(
			x=Nlives, y=sequence, yerr=sequence_errs,
			marker='o', label=str(d)
		)
		color = p[0].get_color()
		del p
		plt.ylabel('R')
		plt.xlabel('Live points')
		plt.xscale('log')
		plt.yscale('log')
		# volume scales as r^d, 1/K is the Poisson Point Process intensity
		# V=r^d ~ 20/K
		# Lambda = 1/20 = (1/K) / r^d ==> r = (1/20)^(1/d)
		plt.plot(Nlives, (d/2)**0.1 * (20 / np.array(Nlives))**(1. / d),
			ls='--', color=color)
		plt.ylim(0.04, 1.5)
		plt.xlim(20, None)
		plt.legend(loc='best', title='Dimensionality')
		plt.savefig('scaling_ellipsoids.pdf', bbox_inches='tight')
	plt.close()
	np.savetxt('ellipsoidnn.txt', outputs)

if __name__ == '__main__':
	evaluate_all(
		ds = [2, 4, 8, 16, 32, 64, 128],
		Nlives=[40, 100, 400, 1000,]
	)
	evaluate_all(
		ds = [2, 4, 8, 16, 32, 64, 128],
		Nlives=[40, 60, 100, 200, 400, 600, 1000, 2000, 4000, 10000]
	)
