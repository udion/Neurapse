import numpy as np
import random

class RANDOM_SPIKE_TRAIN():
    def __init__(self, T, delta_t, lamb, n_out=1):
        self.T = T
        self.delta_t = delta_t
        self.lamb = lamb
        self.n_out = n_out
        self.n_t = int(T/delta_t)
        self.V_train = np.zeros(shape=(self.n_out, self.n_t+1)) # t=0 included
        self.spike_instants = [None]*self.n_out #to hold the spike time instants of each neuron
        self.generate()
    
    def generate(self):
        '''
        spike_instants = list of length n_out each value being arr(time_instants_of_spikes)
        '''
        for i in range(self.n_out):
            spike_instants = random.sample(range(self.n_t+1), 10)
            self.V_train[i, spike_instants] = 1
        for i in range(self.n_out):
            self.spike_instants[i] = np.where(self.V_train[i,:] == 1.0)[0]

'''
usage RANDOM_SPIKE_TRAIN:

import matplotlib.pyplot as plt

ST = RANDOM_SPIKE_TRAIN(T=500*(10**-3), delta_t=0.1*(10**-3), lamb=10).V_train
V = ST.V_train

T = 500*(10**-3)
delta_t = 0.1*(10**-3)
n_t = int(T/delta_t)

plt.plot(list(range(n_t+1)), V[0,:])
plt.xlabel('time')
plt.ylabel('V')
plt.title('Random spike train')
plt.show()
'''

class POISSON_SPIKE_TRAIN():
    def __init__(self, T, delta_t, lamb, n_out=1):
        self.T = T
        self.delta_t = delta_t
        self.lamb = lamb
        self.n_out = n_out
        self.n_t = int(T/delta_t)
        self.V_train = np.zeros(shape=(self.n_out, self.n_t+1)) # t=0 included
        self.spike_instants = [None]*self.n_out #to hold the spike time instants of each neuron
        self.generate()
    
    def generate(self):
        '''
        spike_instants = list of length n_out each value being arr(time_instants_of_spikes)
        '''
        self.V_train = np.random.rand(self.n_out, self.n_t+1)
        self.V_train = self.V_train < self.lamb*self.delta_t
        for i in range(self.n_out):
            self.spike_instants[i] = np.where(self.V_train[i,:] == 1.0)[0]
        # print(self.spike_instants)

def get_spike_instants_from_neuron(V, V_thresh):
    '''
    V : num_neurons X n_t
    '''
    num_neurons, n_t = V.shape
    spike_instants = [None]*num_neurons
    for i in range(num_neurons):
        spike_instants[i] = np.where(V[i,:] > V_thresh)[0]
    return spike_instants



'''
#using POISSON_SPIKE_TRAIN

import matplotlib.pyplot as plt

n_out = 10
T = 500*(10**-3)
delta_t = 0.01*(10**-3)
n_t = int(T/delta_t)

ST = POISSON_SPIKE_TRAIN(T=T, delta_t=delta_t, lamb=100, n_out=n_out)
V = ST.V_train

plt.figure()
plt.suptitle('Posson spike train')
plt.xlabel('time')
plt.ylabel('V')
for i in range(n_out):
    plt.subplot(n_out, 1, i+1)
    plt.plot(list(range(n_t+1)), V[i,:])
plt.show()
'''
