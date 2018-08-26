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

T = 500*(10**-3)
delta_t = 0.1*(10**-3)
n_t = int(T/delta_t)

plt.plot(list(range(n_t+1)), I[0,:])
plt.xlabel('time')
plt.ylabel('I')
plt.title('Random spike train')
plt.show()
