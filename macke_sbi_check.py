import torch
import numpy as np
from sbi import utils
from sbi import analysis
from sbi.inference.base import infer

def	macke_sbi(bounds,n_points, n_simulations = 1000):
	
	prior = utils.BoxUniform(low = bounds[0]*torch.ones(1), high = bounds[1]*torch.ones(1))
	
	def cosine_sim(theta):
		t = torch.linspace(0,2*torch.pi,n_points)
		sim = theta*torch.cos(t)+torch.randn(n_points)
		return sim
	
	posterior = infer(cosine_sim,prior,"SNPE",num_simulations=n_simulations)
	return posterior


def sampling(posterior,observation,bounds,n_pixels,n_samples = 1000):
	samples = posterior.sample((n_samples,), x = observation, show_progress_bars=False)
	log_prob = posterior.log_prob(samples, x = observation)
	counts, bins = np.histogram(samples, bins = n_pixels, range=(bounds[0],bounds[1]), density = True,weights = torch.exp(torch.reshape(log_prob,(n_samples,-1))))
	return counts,bins
	

	
def pairplot(macke_samples):
	plot = analysis.pairplot(macke_samples,limits = [[0.0,10.0]], labels = ['Macke'])[0]
	return plot

	'''
	samples1 = posterior.sample((1,), x = observation)
	log_probability1 = posterior.log_prob(samples1, x = observation)

	samples10 = posterior.sample((10,), x = observation)
	log_probability10 = posterior.log_prob(samples10, x = observation)

	samples100 = posterior.sample((100,), x = observation)
	log_probability100 = posterior.log_prob(samples100, x = observation)

	samples1000 = posterior.sample((1000,), x = observation)
	log_probability1000 = posterior.log_prob(samples1000, x = observation)

	samples10000 = posterior.sample((10000,), x = observation)
	log_probability10000 = posterior.log_prob(samples10000, x = observation)

	print("1: ",samples1[torch.argmax(torch.exp(log_probability1))][0].item())
	print("10: ",samples10[torch.argmax(torch.exp(log_probability10))][0].item())
	print("100: ",samples100[torch.argmax(torch.exp(log_probability100))][0].item())
	print("1000: ",samples1000[torch.argmax(torch.exp(log_probability1000))][0].item())
	print("10000: ",samples10000[torch.argmax(torch.exp(log_probability10000))][0].item())
	print("Param, avg noise: ",45,torch.mean(noise))
	'''
