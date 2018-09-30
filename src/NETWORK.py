import numpy as np
import matplotlib.pyplot as plt
from Synapes import *
from SpikeTrains import *
from LIF import LIF
from CURRENTS import *

class DRN_Const1():
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

class NNetwork_Mixed():
    def __init__(self, Fanout, W, Tau, num_pre, num_post, synapse_type):
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
        self.synapse_type = synapse_type

        # print(self.cell.shape)
        for idx, fanout in enumerate(Fanout):
            for idx1, nid in enumerate(fanout):
                w = W[idx][idx1]
                # print('w:', w)
                I0 = 1*(10**-12)
                tau = 15*(10**-3)
                tau_s = tau/4
                tau_d = Tau[idx][idx1]
                tau_l = 20e-3
                A_up = 0.01
                A_dn = -0.02

                # print(self.synapse_type)

                if self.synapse_type[idx, nid] == 'CONST': #const
                    Sy = CONST_SYNAPSE(w, I0, tau, tau_s, tau_d)
                    self.cell[idx, nid] = (w, tau_d, self.synapse_type[idx,nid], Sy)
                elif self.synapse_type[idx, nid] == 'PLASTIC_B':
                    Sy = PLASTIC_SYNAPSE_B(w, I0, tau, tau_s, tau_d, tau_l, A_up, A_dn)
                    self.cell[idx, nid] = (w, tau_d, self.synapse_type[idx,nid], Sy)    
                
        # print('cell created ', self.cell)

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
        SI = get_spike_instants_from_neuron(self.V_pre_response, self.El)
        # Get currents from synapses for given V_pre_response from pre-neurons
        self.I_post_neuron_list = [np.zeros(shape=(1,self.V_pre_response.shape[1]))]*self.num_post
        self.I_sy_list = [] #this will contain the current for individual synapse
        for i in range(self.num_pre):
            for j in range(self.num_post):
                if self.cell[i,j] != None:
                    Sy = self.cell[i,j][3]
                    I = Sy.getI(self.V_pre_response[i,:].reshape(1, -1), [SI[i]], self.delta_t)
                    self.I_post_neuron_list[j] = self.I_post_neuron_list[j]+I
                    self.I_sy_list.append(I)
        self.I_post = np.array(self.I_post_neuron_list).reshape(self.num_post, -1)

        V0_post = np.ones(shape=(self.num_post,1))
        V0_post = self.El*V0_post
        self.V_post_response = self.all_post_neurons.compute(V0_post, self.I_post, self.delta_t)
        # avg_weights_with_time = np.zeros(shape=(1, self.V_post_response.shape(1)))
        # avg_weights_with_time = [[]]*self.V_post_response.shape[1]
        
        # update the synaptic weights upstream and down stream
        # update the cell as well
        # upstream updates
        for j in range(self.num_post):
            V_cur = self.V_post_response[j,:]
            SI_cur = get_spike_instants_from_neuron(V_cur.reshape(1,-1), self.V_thresh)
            for i in range(self.num_pre):
                if (self.cell[i,j] != None) and (self.cell[i,j][2]=='PLASTIC'):
                    V_pre = self.V_pre_response[i,:]
                    SI_pre = get_spike_instants_from_neuron(V_pre.reshape(1,-1), self.V_thresh)
                    t_d = self.cell[i,j][1]
                    for t_ik in SI_cur[0]:
                        # delay added
                        t_j1 = SI_pre[0]*self.delta_t+t_d
                        # diff
                        t_j2 = t_ik*self.delta_t - t_j1
                        # sorted diff
                        t_j2 = np.array(sorted(t_j2))
                        # print('upstr : ', t_j2)
                        # the closes causal one is with smallest positive diff
                        t_j3 = t_j2[np.where(t_j2>0)]
                        if t_j3 != []:
                            delta_tijk = t_j3[0] #smallest
                            Sy = self.cell[i,j][3]
                            w_new = Sy.weight_update(delta_tijk, 1)
                            self.cell[i,j] = (w_new, self.cell[i,j][1], self.cell[i,j][2], Sy)
                            # return
                        # calculating avg weight of synapses
                        # temp_w = []
                        # for ci in range(self.num_pre):
                        #     for cj in range(self.num_post):
                        #         if self.cell[ci,cj] != None:
                        #             temp_w.append(self.cell[ci,cj][0])
                        # avg_weights_with_time[t_ik] += temp_w

        # downstream updates
        for i in range(self.num_pre):
            V_cur = self.V_post_response[i,:]
            SI_cur = get_spike_instants_from_neuron(V_cur.reshape(1,-1), self.V_thresh)
            for j in range(self.num_post):
                if (self.cell[i,j] != None) and (self.cell[i,j][2]=='PLASTIC'):
                    V_post = self.V_post_response[j,:]
                    SI_post = get_spike_instants_from_neuron(V_post.reshape(1,-1), self.V_thresh)
                    t_d = self.cell[i,j][1]
                    for t_ik in SI_cur[0]:
                        # delay added
                        t_ikd = t_ik*self.delta_t + t_d
                        t_j1 = SI_post[0]*self.delta_t
                        # diff
                        t_j2 = t_ikd - t_j1
                        # sorted diff
                        t_j2 = np.array(sorted(t_j2))
                        # print('dnstr : ', t_j2)
                        # the closes caused one is with smallest negative diff
                        t_j3 = t_j2[np.where(t_j2<0)]
                        if t_j3 != []:
                            delta_tijk = t_j3[-1] #smallest negative diff
                            Sy = self.cell[i,j][3]
                            w_new = Sy.weight_update(delta_tijk, -1)
                            self.cell[i,j] = (w_new, self.cell[i,j][1], self.cell[i,j][2], Sy)
                        # calculating avg weight of synapses
                        # temp_w = []
                        # for ci in range(self.num_pre):
                        #     for cj in range(self.num_post):
                        #         if self.cell[ci,cj] != None:
                        #             temp_w.append(self.cell[ci,cj][0])
                        # avg_weights_with_time[t_ik] += temp_w
        
        # for avg weight plot
        # self.display()
        # avg_weights_t = []
        # for ax1 in avg_weights_with_time:
        #     if ax1 != []:
        #         avg_weights_t.append(np.mean(ax1))
        #     else:
        #         avg_weights_t.append(0)
        # plt.plot(avg_weights_t, 'o')

        return self.V_pre_response, self.V_post_response, self.I_sy_list, self.I_post
    
    def display(self):
        # pre neuron response
        plt.figure(figsize=(25, 15))
        for i in range(self.num_pre):
            plt.plot(self.V_pre_response[i], label='pre-{}'.format(i))
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('V')
        plt.title('response of the pre neurons')
        plt.show()

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
        plt.show()

        #plotting the synaptic input current to the post neurons
        plt.figure(figsize=(25, 15))
        for i in range(self.num_post):
            plt.plot(self.I_post[i,:], label='Ipost-{}'.format(i))
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('I post')
        plt.title('synaptic current input to post neurons')
        plt.show()

        # post neuron response
        plt.figure(figsize=(25, 15))
        for i in range(self.num_post):
            plt.plot(self.V_post_response[i,:100], label='post-{}'.format(i))
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('V')
        plt.title('response of the post neurons')
        plt.show()


class NNetwork_Plastic_B():
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

        # print(self.cell.shape)
        for idx, fanout in enumerate(Fanout):
            for idx1, nid in enumerate(fanout):
                w = W[idx][idx1]
                # print('w:', w)
                I0 = 1*(10**-12)
                tau = 15*(10**-3)
                tau_s = tau/4
                tau_d = Tau[idx][idx1]
                tau_l = 20e-3
                A_up = 0.01
                A_dn = -0.02
                Sy = PLASTIC_SYNAPSE_B(w, I0, tau, tau_s, tau_d, tau_l, A_up, A_dn)
                self.cell[idx, nid] = (w, tau_d, Sy)
        # print('cell created ', self.cell)

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
        SI = get_spike_instants_from_neuron(self.V_pre_response, self.El)
        # Get currents from synapses for given V_pre_response from pre-neurons
        self.I_post_neuron_list = [np.zeros(shape=(1,self.V_pre_response.shape[1]))]*self.num_post
        self.I_sy_list = [] #this will contain the current for individual synapse
        for i in range(self.num_pre):
            for j in range(self.num_post):
                if self.cell[i,j] != None:
                    Sy = self.cell[i,j][2]
                    I = Sy.getI(self.V_pre_response[i,:].reshape(1, -1), [SI[i]], self.delta_t)
                    self.I_post_neuron_list[j] = self.I_post_neuron_list[j]+I
                    self.I_sy_list.append(I)
        self.I_post = np.array(self.I_post_neuron_list).reshape(self.num_post, -1)

        V0_post = np.ones(shape=(self.num_post,1))
        V0_post = self.El*V0_post
        self.V_post_response = self.all_post_neurons.compute(V0_post, self.I_post, self.delta_t)


        # avg_weights_with_time = np.zeros(shape=(1, self.V_post_response.shape(1)))
        # avg_weights_with_time = [[]]*self.V_post_response.shape[1]
        # update the synaptic weights upstream and down stream
        # update the cell as well
        # upstream updates
        for j in range(self.num_post):
            V_cur = self.V_post_response[j,:]
            SI_cur = get_spike_instants_from_neuron(V_cur.reshape(1,-1), self.V_thresh)
            for i in range(self.num_pre):
                if self.cell[i,j] != None:
                    V_pre = self.V_pre_response[i,:]
                    SI_pre = get_spike_instants_from_neuron(V_pre.reshape(1,-1), self.V_thresh)
                    t_d = self.cell[i,j][1]
                    for t_ik in SI_cur[0]:
                        # delay added
                        t_j1 = SI_pre[0]*self.delta_t+t_d
                        # diff
                        t_j2 = t_ik*self.delta_t - t_j1
                        # sorted diff
                        t_j2 = np.array(sorted(t_j2))
                        # print('upstr : ', t_j2)
                        # the closes causal one is with smallest positive diff
                        t_j3 = t_j2[np.where(t_j2>0)]
                        if t_j3 != []:
                            delta_tijk = t_j3[0] #smallest
                            Sy = self.cell[i,j][2]
                            w_new = Sy.weight_update(delta_tijk, 1)
                            self.cell[i,j] = (w_new, self.cell[i,j][1], Sy)
                            # return
                        # calculating avg weight of synapses
                        # temp_w = []
                        # for ci in range(self.num_pre):
                        #     for cj in range(self.num_post):
                        #         if self.cell[ci,cj] != None:
                        #             temp_w.append(self.cell[ci,cj][0])
                        # avg_weights_with_time[t_ik] += temp_w

        # downstream updates
        for i in range(self.num_pre):
            V_cur = self.V_post_response[i,:]
            SI_cur = get_spike_instants_from_neuron(V_cur.reshape(1,-1), self.V_thresh)
            for j in range(self.num_post):
                if self.cell[i,j] != None:
                    V_post = self.V_post_response[j,:]
                    SI_post = get_spike_instants_from_neuron(V_post.reshape(1,-1), self.V_thresh)
                    t_d = self.cell[i,j][1]
                    for t_ik in SI_cur[0]:
                        # delay added
                        t_ikd = t_ik*self.delta_t + t_d
                        t_j1 = SI_post[0]*self.delta_t
                        # diff
                        t_j2 = t_ikd - t_j1
                        # sorted diff
                        t_j2 = np.array(sorted(t_j2))
                        # print('dnstr : ', t_j2)
                        # the closes caused one is with smallest negative diff
                        t_j3 = t_j2[np.where(t_j2<0)]
                        if t_j3 != []:
                            delta_tijk = t_j3[-1] #smallest negative diff
                            Sy = self.cell[i,j][2]
                            w_new = Sy.weight_update(delta_tijk, -1)
                            self.cell[i,j] = (w_new, self.cell[i,j][1], Sy)
                        # calculating avg weight of synapses
                        # temp_w = []
                        # for ci in range(self.num_pre):
                        #     for cj in range(self.num_post):
                        #         if self.cell[ci,cj] != None:
                        #             temp_w.append(self.cell[ci,cj][0])
                        # avg_weights_with_time[t_ik] += temp_w
        # self.display()
        # avg_weights_t = []
        # for ax1 in avg_weights_with_time:
        #     if ax1 != []:
        #         avg_weights_t.append(np.mean(ax1))
        #     else:
        #         avg_weights_t.append(0)
        # plt.plot(avg_weights_t, 'o')

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
        plt.savefig('case{}_plot1.png'.format(case_name))
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
        plt.savefig('case{}_plot2.png'.format(case_name))
        # plt.show()

        #plotting the synaptic input current to the post neurons
        plt.figure(figsize=(25, 15))
        for i in range(self.num_post):
            plt.plot(self.I_post[i,:], label='Ipost-{}'.format(i))
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('I post')
        plt.title('synaptic current input to post neurons')
        plt.savefig('case{}_plot3.png'.format(case_name))
        # plt.show()

        # post neuron response
        plt.figure(figsize=(25, 15))
        for i in range(self.num_post):
            plt.plot(self.V_post_response[i,:], label='post-{}'.format(i))
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('V')
        plt.title('response of the post neurons')
        plt.savefig('case{}_plot4.png'.format(case_name))
        # plt.show()

'''
#usage: 

Fanout = [
    [0,1],
    [0,1],
    [0,1]
]
W = [
    [3000,3000],
    [3000,3000],
    [3000,3000]
]
Tau = [
    [1e-3,8e-3],
    [5e-3,5e-3],
    [9e-3,1e-3]
]  

A = NNetwork_Const(Fanout, W, Tau, 3, 2)

# #case 1
I_pre = np.array([
    50e-9*SQUARE_PULSE(0, 10, 10000).generate(),
    50e-9*SQUARE_PULSE(40, 50, 10000).generate(),
    50e-9*SQUARE_PULSE(80, 90, 10000).generate(),
]).reshape(3,-1)

#case 2
# I_pre = np.array([
#     50e-9*SQUARE_PULSE(70, 80, 10000).generate(),
#     50e-9*SQUARE_PULSE(30, 40, 10000).generate(),
#     50e-9*SQUARE_PULSE(0, 10, 10000).generate(),
# ]).reshape(3,-1)


print(I_pre.shape)
A.compute(I_pre, 1e-4)
A.display(1)
'''


#usage : DRN_const1
def DRN_const1_driver(N, exci_frac, connect_frac):
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

    A = DRN_Const1(Fanout, W, Tau_d=Tau, Tau=15e-3, I0=1e-12, N=N, T=10000, delta_t=1e-4)
    # Creating poisson spike trains as inputs for first 25 neurons
    # Note: Originally this was meant for poisson potential spikes, 
    # here using as current
    T = 10000
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
    T=10000
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

DRN_const1_driver(N=200, exci_frac=0.8, connect_frac=0.1)
