import h5py
import numpy as np
from pycbc.waveform import get_td_waveform
from noise import sample_noise
import random

def sample_observations(length=1):
    # samples observations from a pre-generated noise file and returns it as a tensor of inputed length
    f2 = h5py.File('/home/mrtodd/Todd-Nitz-ProjectSp2023/observations.hdf5','r')
    grp = f2["waveforms"]
    list_of_waveforms = []
    list_of_parameters = []
    for each in grp.keys():
        if length>0:
            dset = grp[each]
            parameters = dset.name
            parameters = parameters[11:].split(',')
            parameters = [float(each) for each in parameters]
            list_of_waveforms.append(dset[:,])
            list_of_parameters.append(parameters)
        length-=1
#     f = open(filename, "r")
#     f1 = open(filename2, "r")
#     list_of_tensors = []
#     parameters = []
#     for each_line in f:
#         list_of_tensors.append(torch.as_tensor(np.fromstring(each_line, sep = ' ')))
#     for each_line in f1:
#         parameters.append(torch.as_tensor(np.fromstring(each_line, sep = ' ')))
    
#     index_list = np.arange(0,length)
#     observations = [list_of_tensors[i] for i in index_list]
#     chosen_parameters = [parameters[i] for i in index_list]
    
    return list_of_waveforms, list_of_parameters


def generate_observations(number = 1000):
    f2 = h5py.File("observations_vinclination.hdf5", "w")
#     f = open("observations_vdistance.txt", "w")
#     f1 = open("observation_parameters_vdistance.txt", "w")
    
    waveforms = f2.create_group("waveforms")
    inclinations = np.linspace(0, 1.5, 10000)
    distances = np.linspace(.1, 50, 10000)

    for i in range(number):
#         inclination = torch.as_tensor(np.random.choice(inclinations))
#         distance = torch.tensor([22.0])
        distance = 1.0
        inclination = np.random.choice(inclinations)
        hp, hc = get_td_waveform(approximant = 'SEOBNRv4', mass1= 200, mass2 = 200,delta_t = 1/4096, f_lower=30)
        noise = sample_noise(len(hp))
        waveform = (noise+hp*.5*(1+np.cos(inclination)**2)/distance)*10**18
        dset = waveforms.create_dataset(str(inclination) + "," + str(distance), data=waveform)
        
#         for each in waveform:
#             f.write(str(each.item()) + ' ')
#         f.write('\n')
#         f1.write(str(inclination.item()) + ' ' + str(distance.item()) + '\n')

#     f.close()
#     f1.close()
    
if __name__ == "__main__":
    generate_observations(1000)
#     wfs, pms = sample_observations(10)
#     print(wfs)
#     print(pms)
    
