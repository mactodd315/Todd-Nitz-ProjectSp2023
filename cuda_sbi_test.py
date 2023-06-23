
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py

from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi import utils
from noise import sample_noise
from observations import sample_observations
from pycbc.waveform import get_td_waveform


bounds = [torch.tensor([0,.1]), torch.tensor([torch.pi/2,50])]
prior = utils.BoxUniform(low = bounds[0]*torch.ones(2), high = bounds[1]*torch.ones(2), device='cuda')


def generate_posterior(prior, num_simulations = 1000, num_workers = 1):
    #     takes in bounds to create prior, then uses n_simulations to train neural network and returns posterior as SBI object
    hp, hc = get_td_waveform(approximant = 'SEOBNRv4', mass1= 200, mass2 = 200,delta_t = 1/4096, f_lower=30)
    signal = torch.as_tensor(hp, device = 'cuda')
    
    def td_approx(param):
        inclination, distance = param
        noise = torch.as_tensor(sample_noise(len(hp), source='cluster'), device = 'cuda')
        simulation = (noise+signal*.5*(1+torch.cos(inclination)**2)/distance)*10**18
        return simulation
    
    simulator, prior = prepare_for_sbi(td_approx, prior)
    inference = SNPE(prior, device = 'cuda')
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=num_simulations)
    density_estimator = inference.append_simulations(theta,x).train()
    posterior = inference.build_posterior(density_estimator)
    return(posterior)

num_simulations = 1000
sbi = generate_posterior(prior,num_simulations)
observations, parameters = sample_observations(3, filename='datafiles/observations.hdf5', source = 'cluster')
samples = sbi.sample((10000,), x=torch.as_tensor(observations[0],device = 'cuda'))
samples = np.asanyarray(samples.cpu())
f=h5py.File("temp_2d_samples.hdf5", 'a')
grp = f.require_group("2d_samples")
dset = grp.create_dataset("samples"+str(num_simulations), data=samples)
dset.attrs["true_parameter"] = parameters[0]
f.close()
