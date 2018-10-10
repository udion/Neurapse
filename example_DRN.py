import numpy as np
import matplotlib.pyplot as plt

import Neurapse
from Neurapse.Synapses import *
from Neurapse.utils.SpikeTrains import *
from Neurapse.Networks import *
from Neurapse.utils.CURRENTS import *




#usage : DRN_Const problem 2 and 3
def DRN_Const_driver(N, exci_frac, connect_frac):
    N_exci = int(N*exci_frac)
    N_inhi = N - N_exci

    # Constructing the network
    exci_neuron_id = range(N_exci)
    inhi_neuron_id = range(N_exci, N)

    Fanout = []
    W = []
    Tau = []
    w0 = 3000
    gamma = 0.77 #for problem 3
    gamma = 1# for problem 2
    for i in exci_neuron_id:
        a = []
        for z in range(N):
            if z!=i:
                a.append(z)
        tempFL = []
        tempWL = []
        tempTL = []
        for j in range(int(connect_frac*N)):
            t = np.random.choice(a)
            tempFL.append(t)
            tempWL.append(gamma*w0)
            tempTL.append(np.random.uniform(1e-3, 20e-3))
        W.append(tempWL)
        Fanout.append(tempFL)
        Tau.append(tempTL)

    for i in inhi_neuron_id:
        tempFL = []
        tempWL = []
        tempTL = []
        for j in range(int(connect_frac*N)):
            t = np.random.choice(exci_neuron_id)
            tempFL.append(t)
            tempWL.append(-w0)
            tempTL.append(1e-3)
        W.append(tempWL)
        Fanout.append(tempFL)
        Tau.append(tempTL)
    print('Fanout: ', Fanout)
    print('W: ', W)
    print('Tau: ', Tau)

    A = DRN_Const(Fanout, W, Tau_d=Tau, Tau=15e-3, I0=1e-12, N=N, T=20000, delta_t=1e-4)
    # Creating poisson spike trains as inputs for first 25 neurons
    # Note: Originally this was meant for poisson potential spikes, 
    # here using as current
    T = 20000
    Tau = 15e-3
    delta_t = 1e-4
    n_out = 25
    I0 = 1e-12
    ST = POISSON_SPIKE_TRAIN(T=int(T*delta_t), delta_t=delta_t, lamb=100, n_out=n_out)
    V_poi_spikes = ST.V_train[:,:-1]
    # print(V_poi_spikes.shape)
    
    reference_alpha = np.zeros(T)
    for t in range(3*int(Tau//delta_t)):
        reference_alpha[t] = np.exp(-t*delta_t/Tau) - np.exp(-4*t*delta_t/Tau)
    # plt.plot(reference_alpha)
    # plt.show()
    
    I_poi = np.zeros(shape=(n_out, T))
    for idx in range(n_out):
        V_sp = V_poi_spikes[idx,:].reshape(1,-1)
        print(V_sp.shape)
        for t,v in enumerate(V_sp[0,:]):
            if v>0:
                t1 = t
                t2 = min(t+int(3*Tau//delta_t), T)
                I_poi[idx, t1:t2] += w0*I0*reference_alpha[0:t2-t1]
    
    # for idx in range(n_out):
    #     plt.plot(I_poi[idx,:])
    #     plt.show()

    I_app = np.zeros(shape=(N, T))
    I_app[0:n_out, :] = I_poi
    El = -70e-3
    V_thresh = 20e-3
    V0 = El*np.ones(shape=(N,1))
    A.compute(I_app=I_app, V0=V0, delta_t=delta_t)

    V_response = A.V_collector
    I_app_feed = A.I_app_collector #same as I_app
    I_synapse_feed = A.I_synapse_feed
    
    exci_spike_instants = get_spike_instants_from_neuron(
        V_response[exci_neuron_id,:],
        V_thresh
    )

    inhi_spike_instants = get_spike_instants_from_neuron(
        V_response[inhi_neuron_id,:],
        V_thresh
    )

    # print(exci_spike_instants)
    # print(inhi_spike_instants)

    colorCodes = np.array(
        [[0,0,0]]*N_exci
        +
        [[0,0,1]]*N_inhi
    )
    plt.eventplot(exci_spike_instants + inhi_spike_instants, color=colorCodes, lineoffsets=2)
    plt.show()

    # for i in range(N):
    #     plt.plot(I_synapse_feed[i, :])
    #     plt.show()
    
    # Ret and Rit
    Ret = []
    Rit = []
    for l in exci_spike_instants:
        Ret = Ret+list(l)
    for l in inhi_spike_instants:
        Rit = Rit+list(l)
    Ret_sorted = sorted(Ret)
    Rit_sorted = sorted(Rit)
    t0 = 100
    plt.figure(figsize=(25, 25))
    plt.subplot(2,1,1)
    plt.hist(Ret_sorted, bins=int(T/t0))
    plt.xlabel('time')
    plt.ylabel('freq Ret')

    plt.subplot(2,1,2)
    plt.hist(Rit_sorted, bins=int(T/t0))
    plt.xlabel('time')
    plt.ylabel('freq Rit')
    plt.show()

#usage : DRN_Plastic problem 4 and 5
def DRN_Plastic_driver(N, exci_frac, connect_frac):
    N_exci = int(N*exci_frac)
    N_inhi = N - N_exci

    # Constructing the network
    exci_neuron_id = range(N_exci)
    inhi_neuron_id = range(N_exci, N)

    Fanout = []
    W = []
    Tau = []

    w0 = 3000
    # gamma = 1 #for problem 4
    gamma = 0.4 # for problem 5
    for i in exci_neuron_id:
        a = []
        for z in range(N):
            if z!=i:
                a.append(z)
        tempFL = []
        tempWL = []
        tempTL = []
        for j in range(int(connect_frac*N)):
            t = np.random.choice(a)
            tempFL.append(t)
            tempWL.append(gamma*w0)
            tempTL.append(np.random.uniform(1e-3, 20e-3))
        W.append(tempWL)
        Fanout.append(tempFL)
        Tau.append(tempTL)

    for i in inhi_neuron_id:
        tempFL = []
        tempWL = []
        tempTL = []
        for j in range(int(connect_frac*N)):
            t = np.random.choice(exci_neuron_id)
            tempFL.append(t)
            tempWL.append(-w0)
            tempTL.append(1e-3)
        W.append(tempWL)
        Fanout.append(tempFL)
        Tau.append(tempTL)
    print('Fanout: ', Fanout)
    print('W: ', W)
    print('Tau: ', Tau)

    A = DRN_Plastic(Fanout, W, Tau_d=Tau, Tau=15e-3, Tau_l=20e-3, I0=1e-12, A_up=0.01, A_dn=-0.07, N=N, N_exci=N_exci, T=20000, delta_t=1e-4, gamma=gamma)
    # Creating poisson spike trains as inputs for first 25 neurons
    # Note: Originally this was meant for poisson potential spikes, 
    # here using as current
    T = 20000
    Tau = 15e-3
    delta_t = 1e-4
    n_out = 25
    I0 = 1e-12
    ST = POISSON_SPIKE_TRAIN(T=int(T*delta_t), delta_t=delta_t, lamb=100, n_out=n_out)
    V_poi_spikes = ST.V_train[:,:-1]
    # print(V_poi_spikes.shape)
    
    reference_alpha = np.zeros(T)
    for t in range(3*int(Tau//delta_t)):
        reference_alpha[t] = np.exp(-t*delta_t/Tau) - np.exp(-4*t*delta_t/Tau)
    # plt.plot(reference_alpha)
    # plt.show()
    
    I_poi = np.zeros(shape=(n_out, T))
    for idx in range(n_out):
        V_sp = V_poi_spikes[idx,:].reshape(1,-1)
        print(V_sp.shape)
        for t,v in enumerate(V_sp[0,:]):
            if v>0:
                t1 = t
                t2 = min(t+int(3*Tau//delta_t), T)
                I_poi[idx, t1:t2] += w0*I0*reference_alpha[0:t2-t1]
    
    # for idx in range(n_out):
    #     plt.plot(I_poi[idx,:])
    #     plt.show()

    I_app = np.zeros(shape=(N, T))
    I_app[0:n_out, :] = I_poi
    El = -70e-3
    V_thresh = 20e-3
    V0 = El*np.ones(shape=(N,1))
    A.compute(I_app=I_app, V0=V0, delta_t=delta_t)

    V_response = A.V_collector
    I_app_feed = A.I_app_collector #same as I_app
    I_synapse_feed = A.I_synapse_feed

    avg_weights = A.avg_weights
    plt.plot(avg_weights)
    plt.xlabel('time steps')
    plt.ylabel('mean excitatory synapse weights')
    plt.show()
    
    exci_spike_instants = get_spike_instants_from_neuron(
        V_response[exci_neuron_id,:],
        V_thresh
    )

    inhi_spike_instants = get_spike_instants_from_neuron(
        V_response[inhi_neuron_id,:],
        V_thresh
    )

    colorCodes = np.array(
        [[0,0,0]]*N_exci
        +
        [[0,0,1]]*N_inhi
    )
    plt.eventplot(exci_spike_instants + inhi_spike_instants, color=colorCodes, lineoffsets=2)
    plt.show()

    # Ret and Rit
    Ret = []
    Rit = []
    for l in exci_spike_instants:
        Ret = Ret+list(l)
    for l in inhi_spike_instants:
        Rit = Rit+list(l)
    Ret_sorted = sorted(Ret)
    Rit_sorted = sorted(Rit)
    t0 = 100
    plt.figure(figsize=(25, 25))
    plt.subplot(2,1,1)
    plt.hist(Ret_sorted, bins=int(T/t0))
    plt.xlabel('time')
    plt.ylabel('freq Ret')

    plt.subplot(2,1,2)
    plt.hist(Rit_sorted, bins=int(T/t0))
    plt.xlabel('time')
    plt.ylabel('freq Rit')
    plt.show()

# DRN_const_driver(N=200, exci_frac=0.8, connect_frac=0.1)
DRN_Plastic_driver(N=200, exci_frac=0.8, connect_frac=0.1)