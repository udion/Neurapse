import numpy as np
import matplotlib.pyplot as plt
from SpikeTrains import POISSON_SPIKE_TRAIN, RANDOM_SPIKE_TRAIN

class CONST_SYNAPSE():
    '''
    This synapsecan be represented
    by a single non changing weight
    '''
    def __init__(self, w, I0, tau, tau_s):
        self.w = w
        self.I0 = I0
        self.tau = tau
        self.tau_s = tau_s
    
    def getI(self, V_train, spike_instants, delta_t):
        '''
        V_train : 1 X n_t
        spike_instants : list(arr(num_of_spikes))
        
        returns It = 1 X n_t
        '''
        n_t = V_train.shape[1]
        self.It = np.zeros(shape=(1,n_t))
        # print(spike_instants)
        # return
        for t in range(n_t):
            contribution = np.array(spike_instants[0])<t
            contribution_i = np.where(contribution == 1)[0]
            t_calc = np.array(spike_instants[0][contribution_i])
            if t_calc != np.array([]):
                s = self.f(t*delta_t, t_calc*delta_t)
                self.It[0, t] = self.I0*self.w*s
            else:
                self.It[0, t] = 0
        return self.It
    
    def f(self, t, t_calc):
        s1 = np.exp(-(t - t_calc)/self.tau)
        s2 = np.exp(-(t - t_calc)/self.tau_s)
        s = s1-s2
        s = np.sum(s)
        return s

    # upd_coeff is {-1,1} according to increment/decrement rule
    def weight_update(self, gamma, delta_tk, upd_coeff):
        '''
        update the weight and will return the delta by which it updated
        '''
        s1 = np.exp(- delta_tk/self.tau)
        s2 = np.exp(- delta_tk/self.tau_s)
        if upd_coeff == -1:
            if self.w <= 1:
                self.w = 1
                return 1-self.w
            else:
                self.w = self.w + upd_coeff*self.w*gamma*(s1 - s2)
                return upd_coeff*self.w*gamma*(s1 - s2)
                # print('weights fixed to 10')
        elif upd_coeff == 1:
            if self.w >= 500:
                self.w = 500
                return 500-self.w
                # print('weights fixed to 500')
            else:
                self.w = self.w + upd_coeff*self.w*gamma*(s1 - s2)
                return upd_coeff*self.w*gamma*(s1 - s2)
        
        

'''
using SYNAPSE and SPIKETRAINS


n_out = 1
T = 500*(10**-3)
delta_t = 0.1*(10**-3)
n_t = int(T/delta_t)

ST = POISSON_SPIKE_TRAIN(T=T, delta_t=delta_t, lamb=10, n_out=n_out)
V, SI = ST.V_train, ST.spike_instants

w = 500
I0 = 1*(10**-12)
tau = 15*(10**-3)
tau_s = tau/4
Sy = CONST_SYNAPSE(w, I0, tau, tau_s)
I = Sy.getI(V, SI, delta_t)

plt.figure()
plt.suptitle('spike train and synaptic current')

plt.subplot(2,1,1)
plt.plot(list(range(n_t+1)), V[0,:])
plt.xlabel('time')
plt.ylabel('V')

plt.subplot(2,1,2)
plt.plot(list(range(n_t+1)), I[0,:])
plt.xlabel('time')
plt.ylabel('I')
plt.show()
'''

