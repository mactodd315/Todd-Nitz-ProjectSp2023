import numpy as np
import matplotlib.pyplot as plt
from simulator import *
import time

# evaluation time goes like np1*np2*nsim
n_prior1 = 1000 # spread phse
n_prior2 = 2000 # zooming phase
n_sim = 500

#generates "x_obs"
A = np.random.ranf()*15
B = np.random.ranf()*10
x_obs,data_obs = simulate(A,B, n = n_sim)
plt.plot(x_obs, data_obs)
plt.show()



#initial guess
B = float(input("Enter Guess for Frequency: "))
start = time.time()

#starts initial spread of values for prior (to be updated later) and compares to wider threshold
threshold = 3
posterior =[]
theta = np.linspace(B/2,3*B/2,n_prior1)
for each in theta:
    x_sim,data_sim = simulate(A, each, n = n_sim,noise = False)
    norm = [abs(data_obs[i]-data_sim[i]) for i in range(n_sim)]
    if np.average(norm) < threshold:
        posterior.append(each)

round1 = time.time()-start
print("Time for Phase 1 Analysis: ", round(round1,3), " seconds")

#samples theta from p(theta), taken from 1st posterior, compares distance to x_obs with narrower threshold 
threshold = .9
new_B = np.average(posterior)
theta = np.random.normal(loc = new_B, size = n_prior2)
posterior = []
for each in theta:
    x_sim,data_sim = simulate(A, each, n = n_sim,noise = False)
    norm = [abs(data_obs[i]-data_sim[i]) for i in range(n_sim)]
    if np.average(norm) < threshold:
        posterior.append(each)
    
round2 = time.time()-start
print("Time for Phase 2 Analysis: ", round(round2,3), " seconds")
print("Number of Accepted Parameters: ",len(posterior))

# plots histogram indicating probable parameter and uses median to fit the curve
fig, (p1,p2) = plt.subplots(1,2,figsize=(12, 6))

p1.hist(posterior)
p1.set_xlabel('B - Frequency')

p2.plot(x_obs,data_obs, label = 'Observed')
fit = np.median(posterior)
x_fit,data_fit = simulate(A,fit, noise = False)
p2.plot(x_fit, data_fit, label = 'Fit -- Paramters: A={:.2f} (fixed), B={:.2f}'.format(A, fit))
p2.legend()

plt.show()