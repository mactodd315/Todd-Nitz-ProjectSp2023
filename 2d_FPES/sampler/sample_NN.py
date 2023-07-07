from sbi.inference import SNPE
import torch, numpy, pickle, h5py, sys
import matplotlib.pyplot as plt

work_folder = '/home/mrtodd/2d_FPES/'
training_number = str(sys.argv[1])
nn_file = open(work_folder+'train_NN/NN'+training_number+'.pickle', 'rb')
neural_net = pickle.load(nn_file)


observation_file = h5py.File(work_folder+'training_data/training_samples'+training_number+'.hdf', 'r')
observation_group = observation_file['2d_dis_inc']
distances = observation_group['distances']
inclinations = observation_group['inclinations']

posteriors_file = h5py.File(work_folder+'sample_nn/posteriors.hdf', 'a')
posteriors_group = posteriors_file.create_group('post_samples:'+training_number+'ts')

bins = 200
observation_number = len(distances)
print(observation_number)

xdataset = posteriors_group.create_dataset('distance_samples', (observation_number,bins))
ydataset = posteriors_group.create_dataset('inclination_samples', (observation_number,bins))
xbinedges = posteriors_group.create_dataset('xbin_edges', (observation_number, bins+1))
ybinedges = posteriors_group.create_dataset('ybin_edges', (observation_number, bins+1))

for index in range(observation_number):
    observation = torch.as_tensor(observation_group['waveforms'][index,:])
    distance = distances[index]
    inclination = inclinations[index]

    samples = neural_net.sample((10000,), x=observation, show_progress_bars = False)
    dist_samples = numpy.array(samples[:,0])
    inc_samples = numpy.array(samples[:,1])
    counts, xbin_edges, ybin_edges = numpy.histogram2d(dist_samples, inc_samples,bins, range=[[10,100], [0,3.14159]], density=True)
    posterior = [counts[:,0]*90/bins, counts[:,1]*3.14159/bins]
    xdataset[index,:] = posterior[0]
    ydataset[index,:] = posterior[1]
    xbinedges[index,:] = xbin_edges
    ybinedges[index,:] = ybin_edges
    
posteriors_file.close()

