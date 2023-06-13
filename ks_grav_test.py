"""
Runs a percentile percentile test for the amplitude/distance gravitational wave waveform parameter estimation from grav_test.py
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
import h5py

from sbi import utils
from sbi import analysis
from sbi.inference.base import infer

from pycbc.waveform import get_td_waveform
from observations import sample_observations
from noise import sample_noise

def generate_posterior(prior, num_simulations = 1000, num_workers = 1):
#     takes in bounds to create prior, then uses n_simulations to train neural network and returns posterior as SBI object
    hp, hc = get_td_waveform(approximant = 'SEOBNRv4', mass1= 200, mass2 = 200,delta_t = 1/4096, f_lower=30)
    signal = torch.as_tensor(hp)
    
    def td_approx(param):
        inclination, distance = param
        noise = torch.as_tensor(sample_noise(len(hp)))
        simulation = (noise+signal*.5*(1+torch.cos(inclination)**2)/distance)*10**18
        return simulation
    
    posterior = infer(td_approx, prior, "SNPE", num_simulations=num_simulations, num_workers=num_workers)
    return posterior

def get_counts(posterior, observation, bounds,n_pixels):
    observation = torch.as_tensor(observation)
    posterior_samples = np.asanyarray(posterior.sample((10000,), x = observation, show_progress_bars = False))
#     inclination_samples = [sample[0] for sample in posterior_samples]
    distance_samples = [sample[1] for sample in posterior_samples]
    print(distance_samples)
    counts, bins = np.histogram(distance_samples, bins = n_pixels, range=(bounds[0],bounds[1]), density = True)
    print(counts)
    return counts

def true_in_interval(interval_width, theta, counts, true_parameter):
#     calculates the range of parameters that correspond to percentiles
    credibility_bounds = theta[np.searchsorted(counts.cumsum(),[0,interval_width])]
    if true_parameter <= credibility_bounds[1].item() and true_parameter >= credibility_bounds[0].item():
        return True
    else:
        return False
    


if __name__ == "__main__":
#     n_observations = 1 
    num_simulations = 10000
    num_workers = 1
    n_pixels = 50
                       
    #generating sbi posterior
    
    #1-d problem
    # inclination_bounds = [torch.tensor([0]), torch.tensor([torch.pi/2])]
    # distance_bounds = [torch.tensor([.1]), torch.tensor([50.0])]
    # bounds = distance_bounds
    # prior = utils.BoxUniform(low = bounds[0]*torch.ones(1), high = bounds[1]*torch.ones(1))
    
    #2-d problem
    bounds = [torch.tensor([0,0.1]), torch.tensor([torch.pi/2,50.0])]
    prior = utils.BoxUniform(low = bounds[0]*torch.ones(2), high = bounds[1]*torch.ones(2))
    
    #pass to "posterior" generator
    sbi_posterior = generate_posterior(prior, num_simulations, num_workers)
    print("Posterior Trained.")

    observation, true_parameter = sample_observations()
    print(true_parameter)
    posterior_samples = sbi_posterior.sample((10000,), x = torch.as_tensor(observation))
    _ = analysis.pairplot(posterior_samples, limits = [[0,1.6], [.1,50]], figsize = (5,5))
    plt.show()
    
    #generate trues interval array (y-axis of pp test)
    n_intervals = 10
    credible_intervals = np.linspace(0,1,n_intervals)
    fraction_true_in_interval = np.zeros(n_intervals)
    
    #distance parameter
    print("Sampling Distance parameters.")
    bounds = [0.1,50.0]
    theta =  np.linspace(bounds[0],bounds[1],n_pixels)
    pixelwidth = (bounds[1]-bounds[0])/n_pixels
    for i in range(n_intervals-1):
        ci = credible_intervals[i]
        samples = 0
        print("Evaluating credible interval: ", ci)
        while samples<2:
            print(samples)
            samples+=1
            observation, true_parameter = sample_observations()
            counts = get_counts(sbi_posterior, observation, bounds, n_pixels)*pixelwidth
            print(true_parameter)
            if true_in_interval(ci, theta, counts, true_parameter[1]):
                break
        fraction_true_in_interval[i] = 1/samples
    print(fraction_true_in_interval)
    Result = kstest(fraction_true_in_interval, "uniform")
    print(Result.pvalue)

    f = h5py.File("ks_test_results.hdf5", "a")

    grp = f.creat_group("pp_tests")


     
    #inclination parameter
#     bounds = [0.0,np.pi/2]
#     theta =  np.linspace(bounds[0],bounds[1],n_pixels)
#     pixelwidth = (bounds[1]-bounds[0])/n_pixels
#     for i in range(n_intervals-1):
#         ci = credible_intervals[i]
#         samples = 0
#         while True:
#             samples+=1
#             observation, true_parameter = sample_observations()
#             counts = get_counts(sbi_posterior, observation, bounds, n_pixels)*pixelwidth
#             if true_in_interval(ci, theta, counts, true_parameter[0]):
#                 break
#         necessary_observations[i] = 1/samples
#     print(necessary_observations)
    
#     observations, true_parameters = sample_observations(n_observations)
#     theta = np.linspace(bounds[0],bounds[1],n_pixels)
#     pixelwidth = (bounds[1].item()-bounds[0].item())/n_pixels
#     counts = [get_counts(sbi_posterior, each, n_observations, bounds,n_pixels)*pixelwidth for each in observations]
#     for i in range(n_intervals-1):
#         ci = credible_intervals[i]
#         trues = 0
#         for j in range(n_observations):
#             true_parameter = true_parameters[j][1].item()
#             if true_in_interval(ci, theta, counts[j], true_parameter):
#                 trues += 1
#         trues_in_intervals[i] = trues/n_observations
#     trues_in_intervals[-1] = 1
    
    #plots ideal versus results of pp_test for calculated posterior    
    
#     Result = kstest(trues_in_intervals, "uniform")
#     print(Result)
#     plt.plot(Result.pvalue())
#     plt.xlabel("Credible Interval")
#     plt.ylabel("pvalue")
#     plt.grid()
#     plt.savefig("kstest.png")
    
    
