import torch, h5py, pickle, sys, numpy

import sbi.utils as utils
from sbi.inference import SNPE
from sbi.utils.user_input_checks import process_prior

import configparser

def get_bounds_from_config(filepath, parameter):
    cp = configparser.ConfigParser()	
    cp.read(filepath)
    param_prior = 'prior-'+parameter
    bounds = [float(cp[param_prior]['min-'+parameter]), float(cp[param_prior]['max-'+parameter])]
    return bounds

def train_NN(prior, num_simulations, training_file, noise_file):
    
    training_groups = list(training_file.keys())
    training_group = training_file[training_groups[0]]
    trainings = numpy.array(training_group['waveforms'][:,:])
    temp_ind = numpy.random.choice(range(len(trainings)-num_simulations))
    trainings = numpy.array(training_group['waveforms'][temp_ind:temp_ind+num_simulations,:])
    noise = noise_file['noise']['noise'][()]

    samples_length = len(trainings[0])
    temp_ind = numpy.random.choice(range(len(noise)-samples_length))
    noise = noise[temp_ind:temp_ind+samples_length]
    training_samples = torch.as_tensor(trainings, dtype=torch.float32,device='cuda')+torch.as_tensor(noise, dtype=torch.float32, device = 'cuda')
    sample_parameters = torch.zeros([len( training_group['distances']),2],dtype = torch.float32, device='cuda')
    parameters = [(training_group['distances'][i], training_group['inclinations'][i]) for i in range(len(training_group['distances']))]
    
    for i in range(len(training_group['distances'])):
        sample_parameters[i,0] = parameters[i][0]
        sample_parameters[i,1] = parameters[i][1]
    prior, _, priorr = process_prior(prior)
    inference = SNPE(prior, device='cuda')
    density_estimator = inference.append_simulations(sample_parameters, training_samples).train()
    neural_net = inference.build_posterior(density_estimator)
    return neural_net


if __name__ == "__main__":

    cp = '/home/mrtodd/2d_FPES/training_data/injection.ini'
    distance_bounds = get_bounds_from_config(cp, 'distance')
    inclination_bounds = get_bounds_from_config(cp, 'inclination')
    bounds = [torch.tensor([distance_bounds[0],inclination_bounds[0]]), torch.tensor([distance_bounds[1],inclination_bounds[1]])]
    prior = utils.BoxUniform(low = bounds[0]*torch.ones(2), high = bounds[1]*torch.ones(2), device='cuda')

    num_simulations = int(sys.argv[1])
    with h5py.File('/home/mrtodd/2d_FPES/training_data/training_samples10000.hdf', 'r') as training_file:
        with h5py.File('/home/mrtodd/2d_FPES/training_data/noise.hdf', 'r') as noise_file:
            neural_net = train_NN(prior, num_simulations, training_file, noise_file).cpu()
    with open('/home/mrtodd/2d_FPES/train_NN/NN'+str(num_simulations)+'.pickle', 'wb') as f:
        pickle.dump(neural_net, f)



