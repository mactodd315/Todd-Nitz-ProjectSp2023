import h5py
import numpy as np
from pycbc.waveform import get_td_waveform
from noise import sample_noise
import random

def sample_observations(length=1, filename='observations.hdf5', source='local'):
    # samples observations from a pre-generated noise file and returns it as a tensor of inputed length
    if source=='cluster':
        filename = 'home/mrtodd/Todd-Nitz-ProjectSp2023/datafiles'+filename
    f2 = h5py.File(filename,'r')
    grp = f2["waveforms"]
    list_of_keys = []
    list_of_waveforms = []
    list_of_parameters = []
    for each in grp.keys():
        list_of_keys.append(each)
    for i in range(length):
        key = random.choice(list_of_keys)
        dset = grp[key]
        parameters = dset.name
        parameters = parameters[11:].split(',')
        parameters = [float(each) for each in parameters]
        waveform = dset[:,]
        list_of_waveforms.append(waveform)
        list_of_parameters.append(parameters)
       
    return list_of_waveforms, list_of_parameters


def generate_observations(number = 1000):
    f2 = h5py.File("observations_vdistance.hdf5", "w")
#     f = open("observations_vdistance.txt", "w")
#     f1 = open("observation_parameters_vdistance.txt", "w")
    
    waveforms = f2.create_group("waveforms")
    inclinations = np.linspace(0, 1.5, number)
    distances = np.linspace(.1, 50, number)

    for i in range(number):
#         inclination = torch.as_tensor(np.random.choice(inclinations))
#         distance = torch.tensor([22.0])
        distance = distances[i]
        inclination = 1.0
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
    
