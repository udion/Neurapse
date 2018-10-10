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

class DRN_Const():
    def __init__(self, Fanout, W, Tau_d, Tau, I0, N, T, delta_t):
        '''
        Fanout/W/Tau : list of lists representing a single pre-neuron's config
        '''
        self.Fanout = Fanout
        self.W = W
        self.Tau_d = Tau_d #Tau_delays of difffernt synapse connections
        self.Tau = Tau
        print(self.Tau)
        self.I0 = I0
        self.N = N
        self.cell = np.array([[None]*N]*N)

        # Define neurons
        self.C = 300e-12
        self.gL = 30e-9
        self.V_thresh = 20e-3
        self.El = -70e-3
        self.Rp = 2e-3
        self.all_neurons = LIF(self.C, self.gL, self.V_thresh, self.El, self.Rp, num_neurons=N)

        for idx, fanout in enumerate(Fanout):
            for idx1, nid in enumerate(fanout):
                w = self.W[idx][idx1]
                I0 = self.I0
                tau = self.Tau
                tau_s = tau/4
                tau_d = self.Tau_d[idx][idx1]
                Sy = CONST_SYNAPSE(w, I0, tau, tau_s, tau_d)
                self.cell[idx, nid] = (w, tau_d, Sy)

        # params which might be of help when using single compute
        # this will be used with DRN and hence nump_pre and num_post is N
        self.cur_t = 0
        self.T = T #number of timesteps
        self.delta_t = delta_t
        
        # stores the synaptic current feeding INTO post synaptic neuron idx
        self.I_synapse_feed = np.zeros(shape=(self.N, self.T))

        self.reference_alpha = np.zeros(self.T)
        for t in range(3*int(self.Tau//self.delta_t)):
            self.reference_alpha[t] = np.exp(-t*self.delta_t/self.Tau) - np.exp(-4*t*self.delta_t/self.Tau)

    def single_compute(self, V_inst, I_inst, delta_t):
        '''
        V_inst : 2d numpy array containing V input for neurons at instant t [num_neurons X 1]
        I_inst : 2d numpy array containing I input for neurons at instant t [num_neurons X 1]
        self.cur_t : current instant
        self.reference_alphas : shape of reference alpha functions
        self.T : total number of simulation timestep
        '''
        self.delta_t = delta_t
        V_next_inst = self.all_neurons.update_fn(V_inst, I_inst, delta_t)
        #check if V_next_inst spiked
        for nid, V_n in enumerate(V_next_inst[:,0]):
            if V_n > self.V_thresh:
                # need to update corresponding synaptic current
                for idx in range(self.N):
                    if self.cell[nid, idx] != None:
                        tau_d = self.cell[nid, idx][1]
                        syn_weight = self.cell[nid, idx][0]
                        # #without considering axonal delay
                        # t1 = self.cur_t+1
                        # t2 = min(self.cur_t+1+int(3*(self.Tau//self.delta_t)), self.T)
                        
                        #with axonal delay
                        # print('i',nid, idx)
                        t1 = min(self.cur_t+int(tau_d//self.delta_t), self.T)
                        t2 = min(self.cur_t+int(tau_d)+int(3*(self.Tau//self.delta_t)), 
                                self.T
                        )
                        # print(t1, t2)
                        # current_shape = syn_weight*self.I0*self.reference_alpha[(nid,idx)][:t2-t1]
                        current_shape = syn_weight*self.I0*self.reference_alpha[:t2-t1]
                        self.I_synapse_feed[idx, t1:t2] += current_shape
        self.cur_t = self.cur_t+1
        return V_next_inst

    def compute(self, I_app, V0, delta_t):
        '''
        I_app : 2d numpy array containing applied I for neurons [N X T]
        V0 : Intital V for all the neurons [N X 1]
        '''
        self.delta_t = delta_t
        self.V_collector = np.zeros(shape=(self.N, self.T))
        self.I_app_collector = I_app
        V_inst = V0
        for t in range(self.T):
            I_inst = I_app[:,t].reshape(-1, 1)
            I_inst += self.I_synapse_feed[:,t].reshape(-1,1)
            V_next_inst = self.single_compute(V_inst, I_inst, self.delta_t)
            self.V_collector[:,t] = V_inst[:,0]
            #time is also updated
            V_inst = V_next_inst

class DRN_Plastic():
    def __init__(self, Fanout, W, Tau_d, Tau, Tau_l, I0, A_up, A_dn,  N, N_exci, T, delta_t, gamma=1):
        '''
        Fanout/W/Tau : list of lists representing a single pre-neuron's config
        Order is such that neurins 0 to N_exci-1 are excitatory, rest inhibitory
        '''
        self.Fanout = Fanout
        self.W = W
        self.Tau_d = Tau_d #Tau_delays of difffernt synapse connections
        self.Tau = Tau
        self.Tau_l = Tau_l
        self.A_up = A_up
        self.A_dn = A_dn
        self.gamma = gamma
        print(self.Tau)
        self.I0 = I0
        self.N = N
        self.N_exci = N_exci
        self.cell = np.array([[None]*N]*N)

        # Define neurons
        self.C = 300e-12
        self.gL = 30e-9
        self.V_thresh = 20e-3
        self.El = -70e-3
        self.Rp = 2e-3
        self.all_neurons = LIF(self.C, self.gL, self.V_thresh, self.El, self.Rp, num_neurons=N)
        self.avg_weights = []

        for idx, fanout in enumerate(Fanout):
            for idx1, nid in enumerate(fanout):
                w = self.W[idx][idx1]
                I0 = self.I0
                tau = self.Tau
                tau_s = tau/4
                tau_d = self.Tau_d[idx][idx1]
                Sy = CONST_SYNAPSE(w, I0, tau, tau_s, tau_d)
                self.cell[idx, nid] = (w, tau_d, Sy)

        # params which might be of help when using single compute
        # this will be used with DRN and hence nump_pre and num_post is N
        self.cur_t = 0
        self.T = T #number of timesteps
        self.delta_t = delta_t
        
        # stores the synaptic current feeding INTO post synaptic neuron idx
        self.I_synapse_feed = np.zeros(shape=(self.N, self.T))

        self.reference_alpha = np.zeros(self.T)
        for t in range(3*int(self.Tau//self.delta_t)):
            self.reference_alpha[t] = np.exp(-t*self.delta_t/self.Tau) - np.exp(-4*t*self.delta_t/self.Tau)
        
        self.downstream_compute = {} #(neuron1, t_inst, neuron2)

    def single_compute(self, V_inst, I_inst, delta_t):
        '''
        V_inst : 2d numpy array containing V input for neurons at instant t [num_neurons X 1]
        I_inst : 2d numpy array containing I input for neurons at instant t [num_neurons X 1]
        self.cur_t : current instant
        self.reference_alphas : shape of reference alpha functions
        self.T : total number of simulation timestep
        '''
        self.delta_t = delta_t
        V_next_inst = self.all_neurons.update_fn(V_inst, I_inst, delta_t)
        #check if V_next_inst spiked
        for nid, V_n in enumerate(V_next_inst[:,0]):
            if V_n > self.V_thresh:
                # need to update corresponding synaptic current
                for idx in range(self.N):
                    if self.cell[nid, idx] != None:
                        tau_d = self.cell[nid, idx][1]
                        syn_weight = self.cell[nid, idx][0]
                        # #without considering axonal delay
                        # t1 = self.cur_t+1
                        # t2 = min(self.cur_t+1+int(3*(self.Tau//self.delta_t)), self.T)
                        
                        #with axonal delay
                        # print('i',nid, idx)
                        t1 = min(self.cur_t+int(tau_d//self.delta_t), self.T)
                        t2 = min(self.cur_t+int(tau_d)+int(3*(self.Tau//self.delta_t)), 
                                self.T
                        )
                        # print(t1, t2)
                        # current_shape = syn_weight*self.I0*self.reference_alpha[(nid,idx)][:t2-t1]
                        current_shape = syn_weight*self.I0*self.reference_alpha[:t2-t1]
                        self.I_synapse_feed[idx, t1:t2] += current_shape
        
        # do upstream weight adjust
        for nid, V_n in enumerate(V_next_inst[:,0]):
            if V_n > self.V_thresh:
                for idx in range(self.N_exci): #only the excitator are plastic
                    if self.cell[idx, nid] != None:
                        tau_d = self.cell[idx, nid][1]
                        w_old = self.cell[idx, nid][0]
                        t_ik = self.cur_t
                        temp_1 = np.where(self.V_collector[idx,:(t_ik-int(tau_d//self.delta_t))]>self.V_thresh)[0]
                        if temp_1.size != 0:
                            t_jlast = (temp_1 + int(tau_d//self.delta_t))[-1]
                            w_new = w_old + w_old*self.A_up*self.gamma*np.exp(-(t_ik - t_jlast)*self.delta_t/self.Tau_l)
                            I0 = self.I0
                            tau = self.Tau
                            tau_s = tau/4
                            Sy_new = CONST_SYNAPSE(w_new, self.I0, tau, tau_s, tau_d)
                            self.cell[idx, nid] = (w_new, tau_d, Sy_new)
        
        # do downstream weight adjust
        for nid, V_n in enumerate(V_next_inst[:,0]):
            if (V_n > self.V_thresh) and (nid in range(self.N_exci)):
                for idx in self.Fanout[nid]:
                    tau_d = self.cell[nid, idx][1]
                    w_old = self.cell[nid, idx][0]
                    t_ik = self.cur_t
                    delayed_t = t_ik + int(tau_d//tau_d)
                    V_post = self.V_collector[idx, :self.cur_t].reshape(1, -1)
                    SI_post = get_spike_instants_from_neuron(V_post, self.V_thresh)
                    # print('sipost ', SI_post)
                    if SI_post[0].size != 0:
                        t_jlast = SI_post[0][0]
                        # print(t_jlast)
                        w_new = w_old + w_old*self.A_dn*self.gamma*np.exp(-(delayed_t - t_jlast)*self.delta_t/self.Tau_l)
                        I0 = self.I0
                        tau = self.Tau
                        tau_s = tau/4
                        Sy_new = CONST_SYNAPSE(w_new, I0, tau, tau_s, tau_d)
                        self.cell[nid, idx] = (w_new, tau_d, Sy_new)

        # getting avg weight at this instant
        w_temp = []
        for idx1 in range(self.N_exci):
            for idx2 in self.Fanout[idx1]:
                w_temp.append(self.cell[idx1, idx2][0])
        self.avg_weights.append(np.mean(w_temp))
        self.cur_t = self.cur_t+1
        return V_next_inst

    def compute(self, I_app, V0, delta_t):
        '''
        I_app : 2d numpy array containing applied I for neurons [N X T]
        V0 : Intital V for all the neurons [N X 1]
        '''
        self.delta_t = delta_t
        self.V_collector = np.zeros(shape=(self.N, self.T))
        self.I_app_collector = I_app
        V_inst = V0
        for t in range(self.T):
            I_inst = I_app[:,t].reshape(-1, 1)
            I_inst += self.I_synapse_feed[:,t].reshape(-1,1)
            V_next_inst = self.single_compute(V_inst, I_inst, self.delta_t)
            self.V_collector[:,t] = V_inst[:,0]
            #time is also updated
            V_inst = V_next_inst
            if t%1000 == 0:
                print('{} step done'.format(t))