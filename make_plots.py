"""
Input .hdf5 file and plot output data
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest

linestyle_dict = {"1000": ":","10000": "-.", "100000": "--", "1000000": "-"}
color_dict1 = {'inclination':'g', 'distance':'b'}
hist_color_dict = {"100": "y","1000": "b","10000": "r", "100000": "g"}
alpha_dict = {"100": .25,"1000": .50,"10000": .75, "100000": 1}

# def annotate_pvalue(credible_interval,dset):
#     y_value = dset[round(dset.size/2)]
#     x_value = credible_interval[round(dset.size/2)]
#     plt.annotate("{:.3e}".format(kstest(dset,"uniform").pvalue),xy=(x_value,y_value),
#                  xycoords = 'data', xytext = (-15,-15), textcoords = 'offset points',
#                  arrowprops = {'arrowstyle': 'simple'})


def makeplots(dsets):
    group = dsets[0].parent.name
    filename = dsets[0].file.filename
    plt.figure(figsize=(10,6))
    if group == "/pp_tests":
        for each_dset in dsets:
            n_intervals = each_dset.size
            credible_interval = np.linspace(0,1,n_intervals)
            dset = each_dset[:,]
            sample_parameter = each_dset.attrs['sample parameter']
            n_simulations = str(each_dset.attrs['N_Simulations'])
            pvalue = "{:.2e}".format(kstest(dset,"uniform").pvalue)
            plt.plot(credible_interval,dset, color = color_dict1[sample_parameter],
                ls=linestyle_dict[n_simulations], label = n_simulations+", pval: "+pvalue)
        plt.legend()
        plt.grid()
        plt.savefig('2param_pptest.png')
    elif filename=='datafiles/1d_training.hdf5':
        for each_dset in dsets:
            samples = each_dset[:,]
            parameters = each_dset.attrs.get('params') 
            print(parameters)
            plt.hist(samples, bins=1000, range=(0,np.pi/2),density=True,
                    color = hist_color_dict[str(each_dset.attrs['TS'])],
                    alpha = alpha_dict[str(each_dset.attrs['TS'])],
                    label = str(each_dset.attrs.get('TS')))
            plt.vlines(parameters[0],0,10, colors='k')
        plt.legend()
        plt.title((dsets[0].parent.name))
        plt.show()  


if __name__ =="__main__":
    # file = h5py.File(input("Input .hdf5 file: "), 'r')
    file = h5py.File('datafiles/1d_training.hdf5', 'r')
    print("Groups: ", file.keys())
    # group = file[input("Which group would you like to access: ")]
    group = file['inclination']
    print("Datasets: ", group.keys())
    #dsets = [group[dset] for dset in input("Which dataset(s) would you like to access? Seperate by commas.\n").split(',')]
    dsets = [group[name] for name in group.keys()]
    makeplots(dsets)
    
