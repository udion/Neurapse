import numpy as np
import matplotlib.pyplot as plt
import random

class RANDOM_SPIKE_TRAIN():
    def __init__(self, T, delta_t, lamb, n_out=1):
        self.T = T
        self.delta_t = delta_t
        self.lamb = lamb
        self.n_out = n_out
        self.n_t = int(T/delta_t)
        self.I_train = np.zeros(shape=(self.n_out, self.n_t+1)) # t=0 included
        self.generate()
    
    def generate(self):
        for i in range(self.n_out):
            spike_instants = random.sample(range(self.n_t+1), 10)
            self.I_train[i, spike_instants] = 1

I = RANDOM_SPIKE_TRAIN(T=500*(10**-3), delta_t=0.1*(10**-3), lamb=10)
I = I.I_train

# T = 500*(10**-3)
# delta_t = 0.1*(10**-3)
# n_t = int(T/delta_t)

# plt.plot(list(range(n_t+1)), I[0,:])
# plt.xlabel('time')
# plt.ylabel('I')
# plt.title('Random spike train')
# plt.show()

class POISSON_SPIKE_TRAIN():
    def __init__(self, T, delta_t, lamb, n_out=1):
        self.T = T
        self.delta_t = delta_t
        self.lamb = lamb
        self.n_out = n_out
        self.n_t = int(T/delta_t)
        self.I_train = np.zeros(shape=(self.n_out, self.n_t+1)) # t=0 included
        self.generate()
    
    def generate(self):
        self.I_train = np.random.rand(self.n_out, self.n_t+1)
        self.I_train = self.I_train < self.lamb*self.delta_t
        # for i in range(self.n_out):
        #     spike_instants = random.sample(range(self.n_t+1), 10)
        #     self.I_train[i, spike_instants] = 1

n_out = 10
T = 500*(10**-3)
delta_t = 0.01*(10**-3)
n_t = int(T/delta_t)

I = POISSON_SPIKE_TRAIN(T=T, delta_t=delta_t, lamb=10, n_out=n_out)
I = I.I_train

plt.figure()
plt.suptitle('Posson spike train')
plt.xlabel('time')
plt.ylabel('I')
for i in range(n_out):
    plt.subplot(n_out, 1, i+1)
    plt.plot(list(range(n_t+1)), I[i,:])
plt.show()