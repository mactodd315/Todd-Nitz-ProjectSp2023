"""
Generates noise from pycbc.noise and writes to noise.txt
"""
import h5py
# import torch
import numpy as np
import pycbc.psd
import pycbc.noise

def sample_noise(length):
    # samples noise from a pre-generated noise file and returns it as a tensor of inputed length
    f = h5py.File('noise.hdf5','r')
    grp = f["noise"]
    dset = grp["noise"]
    noise = dset[0:length,]
    
#     listed_contents = contents.split(' ')
#     index = np.random.choice(len(listed_contents)-length)
#     chosen_list = listed_contents[index:index+length]
#     noise = [float(each) for each in chosen_list]
    return noise

def generate_noise():
    f = h5py.File('noise.hdf5','w')

    flow = 30.0
    delta_f = 1.0/16
    flen = int(2048/delta_f) + 1
    psd = pycbc.psd.aLIGOZeroDetHighPower(flen,delta_f,flow)
    delta_t = 1/4096
    t_samples = int(64/delta_t)

    ts = pycbc.noise.noise_from_psd(t_samples,delta_t,psd,seed=127)
    noise = f.create_group("noise")
    dst = noise.create_dataset("noise",data=ts)
    
if __name__ == "__main__":
    generate_noise()
