import torch
import matplotlib.pyplot as plt
import numpy as np

from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi import utils
from noise import sample_noise
from pycbc.waveform import get_td_waveform
import h5py
from observations import sample_observations


def generate_posterior(prior, num_simulations = 1000, num_workers = 1):
    #     takes in bounds to create prior, then uses n_simulations to train neural network and returns posterior as SBI object
    hp, hc = get_td_waveform(approximant = 'SEOBNRv4', mass1= 200, mass2 = 200,delta_t = 1/4096, f_lower=30)
    signal = torch.as_tensor(hp)
    
    def td_approx(param):
        distance = torch.tensor(1.0)
        inclination = param
        noise = torch.as_tensor(sample_noise(len(hp)))
        simulation = (noise+signal*.5*(1+torch.cos(inclination)**2)/distance)*10**18
        return simulation

    simulator, prior = prepare_for_sbi(td_approx, prior)
    inference = SNPE(prior)
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=num_simulations)
    density_estimator = inference.append_simulations(theta,x).train()
    posterior = inference.build_posterior(density_estimator)
    return(posterior)

# num_simulations=100000
distance_bounds = [torch.tensor([.1]), torch.tensor([50])]
inclination_bounds = [torch.tensor([0]), torch.tensor([torch.pi/2])]
bounds = inclination_bounds
prior = utils.BoxUniform(low = bounds[0]*torch.ones(1), high = bounds[1]*torch.ones(1))

observations, parameters = sample_observations(3, filename='datafiles/observations_vinclination.hdf5')
simulations = [100,1000,10000,100000]

f = h5py.File('datafiles/1d_training.hdf5', 'a')
grp = f.require_group('inclination')

for i in range(4):
    sbi = generate_posterior(prior, simulations[i])
    for j in range(1):
        j=2
        samples = np.asanyarray(sbi.sample((10000,), x=observations[2]))
        dset = grp["sample "+str(j)+" TS "+str(simulations[i])]
        dset[:,]=samples
        dset.attrs['TS'] = simulations[i]
        dset.attrs['params'] = parameters[j]


f.close()

