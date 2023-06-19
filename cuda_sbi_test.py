
import torch

from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi import utils
from noise import sample_noise
from pycbc.waveform import get_td_waveform

bounds = [torch.tensor([0,.1]), torch.tensor([torch.pi/2,50])]
prior = utils.BoxUniform(low = bounds[0]*torch.ones(2), high = bounds[1]*torch.ones(2), device='cuda')
inference = SNPE(prior, device='cuda')

def generate_posterior(prior, inference, num_simulations = 1000, num_workers = 1):
    #     takes in bounds to create prior, then uses n_simulations to train neural network and returns posterior as SBI object
    hp, hc = get_td_waveform(approximant = 'SEOBNRv4', mass1= 200, mass2 = 200,delta_t = 1/4096, f_lower=30)
    signal = torch.as_tensor(hp)
    
    def td_approx(param):
        inclination, distance = param
        noise = torch.as_tensor(sample_noise(len(hp)))
        simulation = (noise+signal*.5*(1+torch.cos(inclination)**2)/distance)*10**18
        return simulation

    theta, x = simulate_for_sbi(td_approx, proposal=prior, num_simulations=1000)
    density_estimator = inference.append_simulations(theta,x).train()
    posterior = inference.build_posterior(density_estimator)
    return(posterior)

print(generate_posterior(prior, inference))
