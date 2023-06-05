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

bounds = [torch.tensor([.1,0]), torch.tensor([50,torch.pi/2])]
prior = utils.BoxUniform(low = bounds[0]*torch.ones(2), high = bounds[1]*torch.ones(2))

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
hp=hp[0:1000]

noise_range = len(hp)
print(noise_range, len(ts))
indices = range(len(noise)-noise_range)

signal = torch.as_tensor(hp)
def td_approx(param):
	distance,inclination = param
	index = np.random.choice(indices)
	simulation = (noise[index:index+noise_range]+signal*.5*(1+torch.cos(inclination)**2)/distance)*10**18
	return simulation

distance = torch.tensor(22.0)
inclination = torch.tensor(torch.pi/6)
actual = [distance, inclination]
amplitude = .5*(1+torch.cos(inclination)**2)/distance
x = torch.linspace(0,torch.pi/2,1000)
y = .5*(1+torch.cos(x)**2)/amplitude
observation = td_approx(actual)
	
num_simulations = 10000
posterior = infer(td_approx, prior, 'SNPE', num_simulations = num_simulations)


samples = np.asanyarray(posterior.sample((10000,), x = observation))
inc_samples = [sample[1] for sample in samples]
dis_samples = [sample[0] for sample in samples]


fig, ax = plt.subplots(figsize=(8,6))
ax.hist2d(inc_samples,dis_samples, bins = 200, range = [[0,np.pi/2],[0,50]])
ax.plot(x,y, color = 'r', label = 'Amplitude = {:.3e}\nDistance = {:} Mpc\nInclination = {:.3f} rad'.format(amplitude,distance,inclination))
ax.set_xlabel('Inclination Angle (rad)')
ax.set_ylabel('Distance (Mpc)')
ax.set_title('Num Simulations: {:}\nNum Datapoints: {:}'.format(num_simulations,noise_range))
plt.legend()
plt.show()


# fig, ax = analysis.pairplot(samples, limits = [[0,50],[0,torch.pi/2]], figsize = (8,6), labels = ["Distance","Inclination"])
# ax[0,0].axvline(x = actual[0], label = 'actual = {:.2f}'.format(actual[0]))
# ax[0,0].legend()
# ax[1,1].axvline(x = actual[1], label = 'actual = {:.2f}'.format(actual[1]))
# fig.suptitle("Number of Simulations: {:}\nNumber of Datapoints: {:}".format(num_simulations, noise_range))
# ax[1,1].legend()
# fig.savefig('plots/grav_test_8.png')
# plt.show()
