import numpy as np
import matplotlib.pyplot as plt
from Synapes import *
from SpikeTrains import *
from LIF import LIF
from network import *

class dynamic_random_network():
    def __init__(self, N, exci_frac, connect_frac):
        self.N = N
        self.exci_frac = exci_frac
        self.connect_frac = connect_frac
        self.N_exci = N*self.exci_frac
        self.N_inhi = N*(1-self.exci_frac)

        # Define neurons
        C = 300e-12
        gL = 30e-9
        V_thresh = 20e-3
        El = -70e-3
        Rp = 2e-3
        self.all_exci_neurons = LIF(C, gL, V_thresh, El, Rp, num_neurons=self.N_exci)
        self.all_inhi_neurons = LIF(C, gL, V_thresh, El, Rp, num_neurons=self.N_inhi)

        # Constructing the network
        exci_neuron_id = range(self.N_exci)
        inhi_neuron_id = range(self.N_exci, self.N)

        self.cell = np.array([[None]*(self.connect_frac*N)]*self.N)
        self.Fanout = np.array([[None]*(self.connect_frac*N)]*self.N)
        self.W = np.array([[None]*(self.connect_frac*N)]*self.N)
        self.Tau = np.array([[None]*(self.connect_frac*N)]*self.N)

        w0 = 3000
        for i in exci_neuron_id:
            for j in range(self.connect_frac * self.N):
                self.Fanout[i][j] = np.random.choice(range(N))
                self.W[i][j] = w0
                self.Tau[i][j] = np.random.uniform(1e-3, 20e-3)

        for i in inhi_neuron_id:
            for j in range(self.connect_frac * self.N):
                self.Fanout[i][j] = np.random.choice(exci_neuron_id)
                self.W[i][j] = -w0
                self.Tau[i][j] = 1e-3

        # Dynamic_random_network = NeuronalNetwork(Fanout, W, Tau, self.N, self.N)

        for idx, fanout in enumerate(self.Fanout):
            for idx1, nid in enumerate(fanout):
                w = W[idx][idx1]
                I0 = 1*(10**-12)
                tau = 15*(10**-3)
                tau_s = tau/4
                tau_d = Tau[idx][idx1]
                Sy = CONST_SYNAPSE(w, I0, tau, tau_s, tau_d)
                self.cell[idx, nid] = (w, tau_d, Sy)

        # Forming the weights of the synapses
        # self.W = np.array([1e-3]*self.N)
        # for i in exci_neuron_id:
        #     W[i] = np.random.uniform(1e-3, 20e-3)

