import numpy as np
import matplotlib.pyplot as plt
from Synapes import *
from SpikeTrains import *
from AEF import AEF

'''
problem 1
'''
# n_out = 1
# T = 500*(10**-3)
# delta_t = 0.1*(10**-3)
# n_t = int(T/delta_t)

# ST = POISSON_SPIKE_TRAIN(T=T, delta_t=delta_t, lamb=10, n_out=n_out)
# V, SI = ST.V_train, ST.spike_instants
# V_pre = V

# w = 500
# I0 = 1*(10**-12)
# tau = 15*(10**-3)
# tau_s = tau/4
# Sy = CONST_SYNAPSE(w, I0, tau, tau_s)
# I = Sy.getI(V, SI, delta_t)

# # simulating AEF RH neuron with the above synaptic current as input
# ######## RS model ############
# C1 = 200*(10**-12)
# gl1 = 10*(10**-9)
# El1 = -70*(10**-3)
# Vt1 = -50*(10**-3)
# Delt1 = 2*(10**-3)
# a1 = 2*(10**-9)
# tw1 = 30*(10**-3)
# b1 = 0*(10**-12)
# Vr1 = -58*(10**-3)

# neuronRHs = AEF(C1, gl1, El1, Vt1, Delt1, a1, tw1, b1, Vr1, num_neurons=1)
# V10 = -0.06999992
# U10 = 1.51338825e-16

# def simulate_neuron(type):
#     if type == 'RH':
#         V0, U0 = V10*np.ones(shape=(1,1)), U10*np.ones(shape=(1,1))
#         neurons = neuronRHs
#     elif type == 'IB':
#         V0, U0 = V20*np.ones(shape=(1,1)), U20*np.ones(shape=(1,1))
#         neurons = neuronIBs
#     elif type == 'CH':
#         V0, U0 = V30*np.ones(shape=(1,1)), U30*np.ones(shape=(1,1))
#         neurons = neuronCHs
#     V, U = neurons.compute(V0, U0, I, delta_t)

#     plt.figure(figsize=(15, 20))
#     plt.subplot(4,1,1)
#     plt.plot(list(range(n_t+1)), V_pre[0,:])
#     plt.xlabel('time')
#     plt.ylabel('V spike train pre-synaptic neuron')

#     plt.subplot(4,1,2)
#     plt.plot(list(range(n_t+1)), I[0,:])
#     plt.xlabel('time')
#     plt.ylabel('I synaptic current')

#     plt.subplot(4,1,3)
#     plt.plot(V[0,:], 'r')
#     plt.ylabel('membrane potential post synaptic neuron')
#     plt.xlabel('time')
#     plt.legend(loc=1)

#     plt.subplot(4,1,4)
#     plt.plot(U[0,:], 'r',)
#     plt.ylabel('U(t) post synaptic neuron')
#     plt.xlabel('time')
#     plt.legend(loc=1)
#     plt.show()

# simulate_neuron('RH')


'''
problem 2
'''
n_out = 100
T = 500*(10**-3)
delta_t = 0.1*(10**-3)
lamb = 1
n_t = int(T/delta_t)

ST = POISSON_SPIKE_TRAIN(T=T, delta_t=delta_t, lamb=lamb, n_out=n_out)
V, SI = ST.V_train, ST.spike_instants
V_pre = V

# for part 1 & 2
w0, sigma_w = 50,5
# w0, sigma_w = 250,25

weights_w = sigma_w*np.random.randn(n_out) + w0
# now I need to get n_out number of synapse with above weights
I0 = 100*(10**-12)
tau = 15*(10**-3)
tau_s = tau/4

Sy_list = []
I_Sy_list = []
for i in range(n_out):
    Sy = CONST_SYNAPSE(weights_w[i], I0, tau, tau_s)
    Sy_list.append(Sy)
    I = Sy.getI(V[i,:].reshape(1,-1), [SI[i]], delta_t)
    I_Sy_list.append(I)

print(len(Sy_list))
print(len(I_Sy_list))

I_total = 0
for I_t in I_Sy_list:
    I_total = I_t+I_total

# simulating the response of the neuron with above current
# simulating AEF RH neuron with the above synaptic current as input
######## RS model ############
C1 = 200*(10**-12)
gl1 = 10*(10**-9)
El1 = -70*(10**-3)
Vt1 = -50*(10**-3)
Delt1 = 2*(10**-3)
a1 = 2*(10**-9)
tw1 = 30*(10**-3)
b1 = 0*(10**-12)
Vr1 = -58*(10**-3)

neuronRHs = AEF(C1, gl1, El1, Vt1, Delt1, a1, tw1, b1, Vr1, num_neurons=1)
V10 = -0.06999992
U10 = 1.51338825e-16

def simulate_neuron(type):
    if type == 'RH':
        V0, U0 = V10*np.ones(shape=(1,1)), U10*np.ones(shape=(1,1))
        neurons = neuronRHs
    elif type == 'IB':
        V0, U0 = V20*np.ones(shape=(1,1)), U20*np.ones(shape=(1,1))
        neurons = neuronIBs
    elif type == 'CH':
        V0, U0 = V30*np.ones(shape=(1,1)), U30*np.ones(shape=(1,1))
        neurons = neuronCHs
    V, U = neurons.compute(V0, U0, I_total, delta_t)

    plt.figure(figsize=(15, 20))

    plt.subplot(3,1,1)
    plt.plot(list(range(n_t+1)), I_total[0,:])
    plt.xlabel('time')
    plt.ylabel('I synaptic current')

    plt.subplot(3,1,2)
    plt.plot(V[0,:], 'r')
    plt.ylabel('membrane potential post synaptic neuron')
    plt.xlabel('time')
    plt.legend(loc=1)

    plt.subplot(3,1,3)
    plt.plot(U[0,:], 'r',)
    plt.ylabel('U(t) post synaptic neuron')
    plt.xlabel('time')
    plt.legend(loc=1)
    plt.show()

simulate_neuron('RH')
