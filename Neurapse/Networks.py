import numpy as np
import matplotlib.pyplot as plt

import Neurapse
from Neurapse.Neurons import LIF
from Neurapse.Synapses import CONST_SYNAPSE
from Neurapse.utils.SpikeTrains import get_spike_instants_from_neuron

class NNetwork_Const():
    def __init__(self, Fanout, W, Tau, num_pre, num_post):
        '''
        Fanout/W/Tau : list of lists representing a single pre-neuron's config
        '''
        self.Fanout = Fanout
        self.W = W
        self.Tau = Tau
        self.num_pre = num_pre
        self.num_post = num_post
        self.cell = np.array([[None]*num_post]*num_pre)

        # Define neurons
        self.C = 300e-12
        self.gL = 30e-9
        self.V_thresh = 20e-3
        self.El = -70e-3
        self.Rp = 2e-3
        self.all_pre_neurons = LIF(self.C, self.gL, self.V_thresh, self.El, self.Rp, num_neurons=num_pre)
        self.all_post_neurons = LIF(self.C, self.gL, self.V_thresh, self.El, self.Rp, num_neurons=num_post)

        for idx, fanout in enumerate(Fanout):
            for idx1, nid in enumerate(fanout):
                w = W[idx][idx1]
                I0 = 1*(10**-12)
                tau = 15*(10**-3)
                tau_s = tau/4
                tau_d = Tau[idx][idx1]
                Sy = CONST_SYNAPSE(w, I0, tau, tau_s, tau_d)
                self.cell[idx, nid] = (w, tau_d, Sy)

    def compute(self, I_pre, delta_t):
        '''
        I_pre : 2d numpy array containing input for pre neurons [num_pre X n_t]
        '''
        self.delta_t = delta_t
        # inital V for LIF neuron is V0 [num_pre X n_t]
        V0_pre = np.ones(shape=(self.num_pre,1))
        V0_pre = self.El*V0_pre
        self.V_pre_response = self.all_pre_neurons.compute(V0_pre, I_pre, self.delta_t)
        # Need spike instants 
        SI = get_spike_instants_from_neuron(self.V_pre_response, self.V_thresh)
        # Get currents from synapses for given V_pre_response from pre-neurons
        self.I_post_neuron_list = [np.zeros(shape=(1,self.V_pre_response.shape[1]))]*self.num_post
        self.I_sy_list = [] #this will contain the current for individual synapse
        for i in range(self.num_pre):
            for j in range(self.num_post):
                if self.cell[i,j] != None:
                    Sy = self.cell[i,j][2]
                    I = Sy.getI(self.V_pre_response[i,:].reshape(1, -1), [SI[i]], self.delta_t)
                    # print(I)
                    self.I_post_neuron_list[j] = self.I_post_neuron_list[j]+I
                    self.I_sy_list.append(I)
        self.I_post = np.array(self.I_post_neuron_list).reshape(self.num_post, -1)
        V0_post = np.ones(shape=(self.num_post,1))
        V0_post = self.El*V0_post
        self.V_post_response = self.all_post_neurons.compute(V0_post, self.I_post, self.delta_t)

        return self.V_pre_response, self.V_post_response, self.I_sy_list, self.I_post
    
    def display(self, case_name=1):
        # pre neuron response
        plt.figure(figsize=(25, 15))
        for i in range(self.num_pre):
            plt.plot(self.V_pre_response[i], label='pre-{}'.format(i))
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('V')
        plt.title('response of the pre neurons')
        plt.savefig('p1_case{}_plot1.png'.format(case_name))
        # plt.show()

        # plotting synaptic current
        plt.figure(figsize=(25, 15))
        n_syn = len(self.I_sy_list)
        print(n_syn)
        for i in range(n_syn):
            plt.plot(self.I_sy_list[i][0,:], label='Sy-{}'.format(i))
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('Synapse I')
        plt.title('synaptic currents')
        plt.savefig('p1_case{}_plot2.png'.format(case_name))
        # plt.show()

        #plotting the synaptic input current to the post neurons
        plt.figure(figsize=(25, 15))
        for i in range(self.num_post):
            plt.plot(self.I_post[i,:], label='Ipost-{}'.format(i))
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('I post')
        plt.title('synaptic current input to post neurons')
        plt.savefig('p1_case{}_plot3.png'.format(case_name))
        # plt.show()

        # post neuron response
        plt.figure(figsize=(25, 15))
        for i in range(self.num_post):
            plt.plot(self.V_post_response[i,:], label='post-{}'.format(i))
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('V')
        plt.title('response of the post neurons')
        plt.savefig('p1_case{}_plot4.png'.format(case_name))
        # plt.show()