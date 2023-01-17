from ast import main
import numpy as np
import matplotlib.pyplot as plt

def simulate(A,B,n=1000, bounds = (0,2*np.pi), noise = True):
    # init data array for specified sine wave with normal noise and unit variance
    (a,b)=bounds
    x = np.linspace(a,b,n)
    data = np.zeros(n)
    if noise:
        for i in range(n):
            data[i] = A*np.sin(B*x[i]) + np.random.normal(scale=1)
    else:
        for i in range(n):
            data[i] = A*np.sin(B*x[i])
    return x,data





#plot
if __name__ == "__main__":
    x,data = simulate(10,1)
    plt.plot(x,data)
    plt.show()
