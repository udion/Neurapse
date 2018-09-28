import numpy as np
import matplotlib.pyplot as plt
from Synapes import *
from SpikeTrains import *
from LIF import LIF
from NETWORK import *

class DRN():
    def __init__(self, N, exci_frac, connect_frac):
        self.N = N
        self.exci_frac = exci_frac
        self.connect_frac = connect_frac
        self.N_exci = int(N*self.exci_frac)
        self.N_inhi = self.N - self.N_exci

        # Define neurons
        C = 300e-12
        gL = 30e-9
        self.V_thresh = 20e-3
        El = -70e-3
        Rp = 2e-3
        self.all_exci_neurons = LIF(C, gL, self.V_thresh, El, Rp, num_neurons=self.N_exci)
        self.all_inhi_neurons = LIF(C, gL, self.V_thresh, El, Rp, num_neurons=self.N_inhi)

        # Constructing the network
        self.exci_neuron_id = range(self.N_exci)
        self.inhi_neuron_id = range(self.N_exci, self.N)

        self.cell = np.array([[None]*int(self.connect_frac*N)]*self.N)
        self.Fanout = []
        self.W = []
        self.Tau = []
        w0 = 3000
        for i in self.exci_neuron_id:
            a = []
            for z in range(N):
                if z!=i:
                    a.append(z)
            tempFL = []
            tempWL = []
            tempTL = []
            for j in range(int(self.connect_frac*self.N)):
                t = np.random.choice(a)
                print(i,j,t)
                tempFL.append(t)
                tempWL.append(w0)
                tempTL.append(np.random.uniform(1e-3, 20e-3))
            self.W.append(tempWL)
            self.Fanout.append(tempFL)
            self.Tau.append(tempTL)

        for i in self.inhi_neuron_id:
            tempFL = []
            tempWL = []
            tempTL = []
            for j in range(int(self.connect_frac * self.N)):
                t = np.random.choice(self.exci_neuron_id)
                tempFL.append(t)
                tempWL.append(-w0)
                tempTL.append(1e-3)
            self.W.append(tempWL)
            self.Fanout.append(tempFL)
            self.Tau.append(tempTL)
        print('Fanout: ', self.Fanout)
        self.compute(5, 1000*(10**-3), 1e-4)
    
    def compute(self, n_out, T, delta_t):
        # Creating poisson spike trains as inputs for first 25 neurons
        # Note: Originally this was meant for poisson potential spikes, 
        # here using as current
        n_t = int(T/delta_t)
        ST = POISSON_SPIKE_TRAIN(T=T, delta_t=delta_t, lamb=100, n_out=n_out)
        I_poi_spikes = ST.V_train

        DRN = NNetwork(self.Fanout, self.W, self.Tau, self.N, self.N)

        I = np.zeros(shape=(self.N, I_poi_spikes.shape[1]))
        I[0:n_out, :] = I_poi_spikes
        self.delta_t = delta_t
        (self.V_pre_response, self.V_post_response, 
        self.I_sy_list, self.I_post) = DRN.compute(I, self.delta_t)

        self.exci_raster = get_spike_instants_from_neuron(
            self.V_post_response[self.exci_neuron_id,:],
            self.V_thresh
        )

        self.inhi_raster = get_spike_instants_from_neuron(
            self.V_post_response[self.inhi_neuron_id,:],
            self.V_thresh
        )

        colorCodes = np.array(
            [[0,0,0]]*self.N_exci
            +
            [[0,0,1]]*self.N_inhi
        )

        print(colorCodes, colorCodes.shape)

        plt.eventplot(self.exci_raster + self.inhi_raster, color=colorCodes, lineoffsets=2)
        plt.show()

        

A = DRN(15, 0.8, 0.4)
