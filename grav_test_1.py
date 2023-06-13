# 
"""
The general structure of this code follows the example sbi code, and is motivated by the desire to apply sbi to infer gravitational wave parameters. The structure is as follows:
-> generates a prior, in this case the parameter is distance
-> defines a simulator (this is already done) and samples from prior to create training data for posterior
-> samples from posterior given an observed data set
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from sbi import utils
from sbi import analysis
from sbi.inference.base import infer
from pycbc.waveform import get_td_waveform
import pycbc.noise
import pycbc.psd

bounds = [0, 10]
prior = utils.BoxUniform(low = bounds[0]*torch.ones(1), high = bounds[1]*torch.ones(1))

flow = 30.0
delta_f = 1.0/16
flen = int(2048/delta_f) + 1
psd = pycbc.psd.aLIGOZeroDetHighPower(flen,delta_f,flow)
delta_t = 1/4096
t_samples = int(32/delta_t)

ts = pycbc.noise.noise_from_psd(t_samples,delta_t,psd,seed=127)
noise = torch.as_tensor(ts)
# [noise.append(i) for i in ts]

hp, hc = get_td_waveform(approximant = 'SEOBNRv4', mass1= 200, mass2 = 200,delta_t = 1/4096, f_lower=30)
signal = torch.as_tensor(hp)
# [signal.append(i) for i in hp]


noise_range = len(hp)
print(noise_range, len(ts))
indices = range(len(noise)-noise_range)
	
def td_approx(distance):
	index = np.random.choice(indices)
	simulation = noise[index:index+noise_range]+signal/distance
	return simulation
	
	
posterior = infer(td_approx,prior,'SNPE', num_simulations=20000)

actual = 2
index = np.random.choice(indices)
observation = noise[index:index+noise_range]+signal/actual

samples = posterior.sample((10000,), x = observation)
_ = analysis.pairplot(samples, limits = [[0,10]], figsize = (8,6))
plt.show()