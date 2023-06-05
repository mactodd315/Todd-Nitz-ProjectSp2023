# percentile-percentile test
# runs a given posterior generating file over n times
# plots percent of true values lying in interval vs. cumulative interval

import torch
import numpy as np
import matplotlib.pyplot as plt

from macke_sbi_check import macke_sbi
from simple_posterior import *

def make_observations(true_p,n_observations,n_points = 1):
	t = np.linspace(0,2*np.pi,n_points)
	observations = [true_p*np.cos(t)+np.random.randn(n_points) for i in range(n_observations)]
	return observations

def sampling(sbi_trained_posterior, observation, n_pixels, bounds, n_samples):
	samples = sbi_trained_posterior.sample((n_samples,), x = observation, show_progress_bars=False)
	#log_prob = sbi_trained_posterior.log_prob(samples, x = observation)
	counts, bins = np.histogram(samples, bins = n_pixels, range=(bounds[0],bounds[1]), density = True)
	#counts, bins = np.histogram(samples, bins = n_pixels, range=(bounds[0],bounds[1]), density = True,weights = torch.exp(torch.reshape(log_prob,(n_samples,-1))))
	return counts
	
def true_in_interval(interval_width,theta,posterior_norm,true_p):
	percent_thetas = theta[np.searchsorted(posterior_norm.cumsum(),[.5-interval_width/2,.5+interval_width/2])]
	if true_p<= percent_thetas[1] and true_p>=percent_thetas[0]:
		return True
	else:
		return False


def percentile_percentile_test(test,true_p, observations, theta, n_pixels, n_posteriors, sbi_posterior = None, n_samples = 10000, n_intervals = 21):
	credible_intervals = np.linspace(0,1,n_intervals)
	trues_in_intervals = np.zeros(n_intervals)
	bounds = [theta[0],theta[-1]]
	pixelwidth = (theta[-1]-theta[0])/n_pixels
	if test == 'analytical':
		analytical_posteriors = [generate_posterior(theta,observed) for observed in observations]
		for i in range(n_intervals-1):
			ci = credible_intervals[i]
			trues_analytical = 0
			for j in range(n_posteriors):
			    if true_in_interval(ci,theta,analytical_posteriors[j],true_p):
			        trues_analytical += 1
			trues_in_intervals[i] = trues_analytical/n_posteriors
		
	if test == 'sbi':
		sbi_samplings = [sampling(sbi_posterior,observed, n_pixels, bounds, n_samples)*pixelwidth for observed in observations]
		for i in range(n_intervals-1):
			ci = credible_intervals[i]
			trues_sbi = 0
			for j in range(n_posteriors):
				if true_in_interval(ci,theta,sbi_samplings[j],true_p):
				    trues_sbi += 1
			trues_in_intervals[i] = trues_sbi/n_posteriors
	trues_in_intervals[-1] = 1
	
	return [credible_intervals,trues_in_intervals]
	
	

true_param = 5.0 # amplitude
	

	
#generate common observation data
n_datapoints= [1,10,100,1000]
n_observations = 100
observations = [make_observations(true_param, n_observations,i) for i in n_datapoints]

n_posteriors = n_observations
bounds = [0,10]
#run posterior generators
n_pixels = 100
theta = np.linspace(bounds[0],bounds[1],n_pixels)
sbi_posteriors = [macke_sbi(bounds,i) for i in n_datapoints]

fig, axs = plt.subplots(1,len(n_datapoints),sharey=True)
for i in range(4):
	pptest = percentile_percentile_test("sbi",true_param, observations[i], theta, n_pixels, n_posteriors, sbi_posterior = sbi_posteriors[i])
	axs[i].plot(pptest[0],pptest[0], label = "Ideal")
	axs[i].plot(pptest[0],pptest[1], label = "sbi: " + str(n_datapoints[i]))
	axs[i].legend()

plt.show()
