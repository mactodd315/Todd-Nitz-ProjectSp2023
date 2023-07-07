import numpy as np
import matplotlib.pyplot as plt
import h5py, sys


def get_posteriors_and_parameters(posterior_file, ts_num, parameter_file, make_2d_plot='no'):
    work_folder = '/home/mrtodd/2d_FPES/'
    posteriors_file = h5py.File(work_folder+posterior_file, 'r')
    print(posteriors_file.keys())
    group = posteriors_file['post_samples:100ts']
    groupkeys = list(group.keys())
    datasets = [group[each] for each in groupkeys]

    if make_2d_plot=='yes':
        f1 = plt.figure()
        X,Y = np.meshgrid(datasets[2],datasets[3])
        H = datasets[0:1]
        print(H.shape)
        print(H)
        plt.pcolormesh(X,Y,H)
        plt.savefig(workfolder+'plots/2dposterior.png')
        plt.clf()

    parameters_file = h5py.File(work_folder+parameter_file, 'r')
    group1 = parameters_file['2d_dis_inc']
    print(group1.keys())
    parameters = np.array(group1['distances'],group1['inclinations'])

    return datasets, parameters
	
def is_true(credibility, cumulative, parameter):
    theta = np.linspace(10,100,200)
    credibility_bounds = theta[np.searchsorted(cumulative,[0,credibility])]
    if parameter<= credibility_bounds[1] and parameter >= credibility_bounds[0]:
        return True
    else:
        return False
		
	
def run_pp_test(dataset, parameters, ts_num):
    n_intervals = 21
    credible_intervals = np.linspace(0,1,n_intervals)
    trues_in_intervals = np.zeros(n_intervals)
    cumulatives = [dataset[j,:].cumsum() for j in range(len(dataset))]
    for i in range(1,n_intervals-1):
        credibility = credible_intervals[i]
        trues_list = [1 if is_true(credibility, cumulatives[j], parameters[j,]) else 0 for j in range(len(dataset)) ]
        trues_in_intervals[i] = sum(trues_list)/len(dataset)
    trues_in_intervals[-1] = 1
    plt.plot(credible_intervals, trues_in_intervals, label = ts_num)	


if __name__ == '__main__':
    for ts_num in sys.argv[1:]:
        ts_num = str(ts_num)
        dataset, parameters = get_posteriors_and_parameters('sample_nn/posteriors.hdf', ts_num, 'training_data/training_samples'+ts_num+'.hdf', 'yes')
        run_pp_test(dataset, parameters, ts_num)
    plt.legend()
    plt.title("Distance Parameter")
    plt.savefig('/home/mrtodd/1d_FPES/plots/pp_test.png')
	
