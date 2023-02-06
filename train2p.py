import numpy as np
import matplotlib.pyplot as plt
from simulator import *
import time

# evaluation time goes like np1*np2*nsim
n_prior1 = 100 # spread phse
n_prior2 = 50000 # zooming phase
n_sim = 500

#generates "x_obs"
A = np.random.ranf()*15
B = np.random.ranf()*10
B = 8
A=12
x_obs,data_obs = simulate(A,B, n = n_sim)
plt.plot(x_obs, data_obs)
plt.show()



#initial guess
A = float(input("Enter Guess for Amplitude: "))
B = float(input("Enter Guess for Frequency: "))
start = time.time()

#starts initial spread of values for prior (to be updated later) and compares to wider threshold
threshold = 3
posterior_A, posterior_B = [],[]
theta_A = np.linspace(A/2,3*A/2,n_prior1)
theta_B = np.linspace(B/2,3*B/2,n_prior1)
for each in theta_A:
    for every in theta_B:
        x_sim,data_sim = simulate(each, every, n = n_sim,noise = False)
        norm = [abs(data_obs[i]-data_sim[i]) for i in range(n_sim)]
        if np.average(norm) < threshold:
            posterior_A.append(each)
            posterior_B.append(every)

round1 = time.time()-start
print("Time for Phase 1 Analysis: ", round(round1,3), " seconds")

#samples theta from p(theta), taken from 1st posterior, compares distance to x_obs with narrower threshold 
threshold = .85
new_A, new_B = np.average(posterior_A), np.average(posterior_B)
posterior_A, posterior_B = [],[]
for i in range(n_prior2):
    theta_A, theta_B = np.random.normal(loc = new_A, scale = .8), np.random.normal(loc = new_B, scale = .8)
    x_sim,data_sim = simulate(theta_A,theta_B, n = n_sim,noise = False)
    norm = [abs(data_obs[i]-data_sim[i]) for i in range(n_sim)]
    if np.average(norm) < threshold:
        posterior_A.append(theta_A)
        posterior_B.append(theta_B)
    
round2 = time.time()-start
print("Time for Phase 2 Analysis: ", round(round2,3), " seconds")
print("Number of Accepted Parameters: ",len(posterior_A))

# plots scatter indicating probable parameters and uses most probable to fit the curve
fig, (p1,p2) = plt.subplots(1,2,figsize=(12, 6))

spread = p1.hist2d(posterior_B,posterior_A, bins = (200,200),range = [[0,10],[0,15]], cmap=plt.cm.jet)
p1.set_xlabel('B - Frequency')
p1.set_ylabel('A - Amplitude')


p2.plot(x_obs,data_obs, label = 'Observed')
A_fit,B_fit = np.median(posterior_A),np.median(posterior_B)
x_fit,data_fit = simulate(A_fit,B_fit, noise = False)
p2.plot(x_fit, data_fit, label = 'Fit -- Paramters: A={:.2f}, B={:.2f}'.format(A_fit,B_fit))
p2.legend()

plt.show()
