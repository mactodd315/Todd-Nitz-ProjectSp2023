"""
Input .hdf5 file and plot output data
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest

linestyle_dict = {"10000": ":", "100000": "--", "1000000": "-"}
color_dict = {'inclination':'g', 'distance':'b'}

def annotate_pvalue(credible_interval,dset):
    y_value = dset[-9]
    x_value = credible_interval[-9]
    plt.annotate("{:.3e}".format(kstest(dset,"uniform").pvalue),xy=(x_value,y_value),
                 xycoords = 'data', xytext = (1.5,-15), textcoords = 'offset points',
                 arrowprops = {'arrowstyle': 'simple'})


def makeplots(dsets):
    group = dsets[0].parent.name
    if group == "/pp_tests":
        n_intervals = dsets[0].size
        credible_interval = np.linspace(0,1,n_intervals)
        for each_dset in dsets:
            dset = each_dset[:,]
            sample_parameter = each_dset.attrs['sample parameter']
            n_simulations = str(each_dset.attrs['N_Simulations'])
            if sample_parameter == 'inclination':
                label = n_simulations
            else:
                label = None
            plt.plot(credible_interval,dset, color = color_dict[sample_parameter],
                ls=linestyle_dict[n_simulations], label = label)
            annotate_pvalue(credible_interval,dset)
        plt.legend()
        plt.grid()
        plt.show()


if __name__ =="__main__":
    #file = h5py.File(input("Input .hdf5 file: "), 'r')
    file = h5py.File('ks_test_results.hdf5', 'r')
    #print("Groups: ", file.keys())
    #group = file[input("Which group would you like to access: ")]
    group = file['pp_tests']
    #print("Datasets: ", group.keys())
    #dsets = [group[dset] for dset in input("Which dataset(s) would you like to access? Seperate by commas.\n").split(',')]
    dsets = [group[name] for name in group.keys()]
    makeplots(dsets)
    