"""
This file generates a posterior for the amplitude estimation of a cosine function
"""


import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt

def generate_posterior(theta, observed):
	# initiallizing
	n_points = observed.size
	t = np.linspace(0,2*np.pi,n_points)
	n_pixels = theta.size

	log_prior = np.zeros(n_pixels)

	log_likelihood = np.ones(n_pixels)

	for i in range(n_pixels):
		sim = theta[i]*np.cos(1*t)
		x = observed - sim
		log_likelihood[i] = np.sum(-x**2/2)
		#for n in range(n_points):
		#	#prob_xn_given_theta = list(np.around(np.random.normal(size = n_draws),1)).count(round(x[n],1))/n_draws
		#	prob_x_given_theta = (2*np.pi)**(-2)*np.exp(-x[n]**2/2)
		#	likelihood[i] *= prob_xn_given_theta
	#print("Done.")
	log_posterior = log_likelihood+log_prior
	norm = logsumexp(log_posterior)
	log_posterior_norm = log_posterior-norm
	#posterior_norm = list(posterior.copy())
	return np.exp(log_posterior_norm)



if __name__ == "__main__":
	# Analysis
	n_pixels = 1000
	true_p = 5
	theta = np.linspace(0,10,n_pixels)
	log_posterior = generate_posterior(true_p,theta)
	theta_est = theta[log_posterior.argmax()]
	print(np.exp(log_posterior[log_posterior.argmax()-50:log_posterior.argmax()+50]))
	print("Analyzing...\n")
	print("Pixel Size: {}".format(10/n_pixels))
	print("Estimated Amplitude: {:}".format(theta_est))
	norm = logsumexp(log_posterior)
	log_posterior_norm = log_posterior-norm
	percent_thetas = theta[np.searchsorted(np.exp(log_posterior_norm).cumsum(),[.25,.75])]
	print(percent_thetas)
	print("Actual Amplitude: {:}\tDifference: {:}".format(true_p, theta_est-true_p))
	#plt.plot(theta,posterior_norm)
	#plt.show()
