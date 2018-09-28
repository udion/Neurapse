import numpy as np
import matplotlib.pyplot as plt
from Synapes import *
from SpikeTrains import *
from LIF import LIF

class NeuronalNetwork():
    def __init__(self, Fanout, W, Tau, num_pre, num_post):
        '''
        Fanout/W/Tau : list of lists representing a single pre-neuron's config
        '''
        self.Fanout = Fanout
        self.W = W
        self.Tau = Tau
        self.cell = np.array([[None]*num_post]*num_pre)

        # Define neurons
        C = 300e-12
        gL = 30e-9
        V_thresh = 20e-3
        El = -70e-3
        Rp = 2e-3
        self.all_pre_neurons = LIF(C, gL, V_thresh, El, Rp, num_neurons=num_pre)
        self.all_post_neurons = LIF(C, gL, V_thresh, El, Rp, num_neurons=num_post)

        # print(self.cell.shape)
        for idx, fanout in enumerate(Fanout):
            for idx1, nid in enumerate(fanout):
                w = W[idx][idx1]
                I0 = 1*(10**-12)
                tau = 15*(10**-3)
                tau_s = tau/4
                tau_d = Tau[idx][idx1]
                Sy = CONST_SYNAPSE(w, I0, tau, tau_s, tau_d)
                self.cell[idx, nid] = (w, tau_d, Sy)
        print(self.cell)


Fanout = [
    [0],
    [0,1],
    [1]
]

W = [
    [3000],
    [3000,3000],
    [3000]
]

Tau = [
    [1],
    [5,5],
    [9]
]      

A = NeuronalNetwork(Fanout, W, Tau, 3, 2)