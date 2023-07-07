from  pycbc.inject import InjectionSet
from pycbc.types import TimeSeries, zeros
import sys, h5py, os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
injfile = sys.argv[1]
with h5py.File(injfile, 'r') as f1:
    distance_list = f1['distance'][()]
    inclination_list = f1['inclination'][()]
    n_distance = len(distance_list)
    injector = InjectionSet(injfile)
    print(distance_list)
    
    with h5py.File('/home/mrtodd/2d_FPES/training_data/training_samples.hdf', 'w') as f:
        twod_group = f.require_group("2d_dis_inc")
        waveforms = twod_group.create_dataset("waveforms", shape=(n_distance,4096))
        distances = twod_group.create_dataset("distances", data=distance_list)
        inclinations = twod_group.create_dataset("inclinations", data=inclination_list)
    
        for i in range(n_distance):
            a = TimeSeries(zeros(4096), epoch=-1.0, delta_t=1.0/1024)
            injector.apply(a, 'H1', simulation_ids=[i])
            waveforms[i,:] = a
    

