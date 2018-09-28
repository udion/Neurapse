import numpy as np
import matplotlib.pyplot as plt
from Synapes import *
from SpikeTrains import *
from AEF import AEF

'''
problem 3 and 4 will use some common functions with same
neurons, hence the following snippet is required by both
'''
n_out = 100
T = 500*(10**-3)
delta_t = 0.1*(10**-3)
lamb = 1
n_t = int(T/delta_t)

ST = POISSON_SPIKE_TRAIN(T=T, delta_t=delta_t, lamb=lamb, n_out=n_out)
V_pre, SI = ST.V_train, ST.spike_instants

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

def simulate_neuron(type, I_total):
    if type == 'RH':
        V0, U0 = V10*np.ones(shape=(1,1)), U10*np.ones(shape=(1,1))
        neurons = neuronRHs
    else:
        print('not supported here!, check AEF file')
        return
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
    # plt.legend(loc=1)

    plt.subplot(3,1,3)
    plt.plot(U[0,:], 'r',)
    plt.ylabel('U(t) post synaptic neuron')
    plt.xlabel('time')
    # plt.legend(loc=1)
    plt.show()
    return V, U

# Now, assuming the initial state is such
# that there are no post synaptic spikes
# then I need to iterate and update weight until
# I get a single post synaptic spike, the following
# snippet will do the same
def iterative_synapse_tune(V_pre, Sy_list, upd_coeff, gamma, max_iter=500):
    '''
    This will update the weights until one spike is seen (if upd_coeff == 1)
    or all spikes die (if upd_coeff == 0) 
    '''
    itr = 0
    del_ws = []
    del_tks = []
    while itr<max_iter:
        mean_del_w = 0
        mean_del_tk = 0
        print('doing iteration {} ...'.format(itr))
        I_Sy_list = []
        for i in range(n_out):
            I = Sy_list[i].getI(V_pre[i,:].reshape(1,-1), [SI[i]], delta_t)
            I_Sy_list.append(I)
        I_total = 0
        for I_t in I_Sy_list:
            I_total = I_t+I_total
        V_post, U_post = simulate_neuron('RH', I_total)
        # check if post neuron spiked, AEF has thrshold of zero
        if upd_coeff == 1:
            if np.sum(V_post[0, :] > 0) > 0:
                print('post neuron already spiked at iter {}'.format(itr))
                print('showing all the weights')
                for z,Sy in enumerate(Sy_list):
                    print('synapse {}: w={}'.format(z, Sy.w))
                break
        elif upd_coeff == -1:
            if np.sum(V_post[0, :] > 0) <= 1:
                print('post neuron spiked only once at iter {}'.format(itr))
                print('showing all the weights')
                for z,Sy in enumerate(Sy_list):
                    print('synapse {}: w={}'.format(z, Sy.w))
                break
        tk_array = np.argwhere(V_post[0,:] == np.amax(V_post))
        tk_array = tk_array.flatten()

        # to find out the right synapse for each of the time
        # instants where post synaptic neuron spiked I will 
        # iterate over all such time instants and to compute 
        # the min delta t between pre-synaptic instants and 
        # current post-synaptic instant
        tk_array_synapse = []
        for tk in tk_array:
            syn_id_list = []
            syn_min_delta_t = []
            for syn_id, si in enumerate(SI):
                if si != []:
                    l = -1*(si-tk)
                    if l[np.where(l>0)] != []:
                        syn_id_list.append(syn_id)
                        syn_min_delta_t.append(np.min(l[np.where(l>0)]))
            if syn_min_delta_t != []:
                min_syn_id = np.argmin(syn_min_delta_t)
                tk_array_synapse.append(( syn_id_list[min_syn_id], delta_t*syn_min_delta_t[min_syn_id] ))

        print(tk_array, len(tk_array))
        print(tk_array_synapse, len(tk_array_synapse))

        for syn_id, syn_delta_t in tk_array_synapse:
            del_w = Sy_list[syn_id].weight_update(gamma, syn_delta_t, upd_coeff)
            mean_del_tk += syn_delta_t
            mean_del_w += del_w
        
        mean_del_tk = mean_del_tk/len(tk_array)
        mean_del_w = mean_del_w/len(tk_array)

        del_tks.append(mean_del_tk)
        del_ws.append(mean_del_w)
        itr = itr+1
    return del_tks, del_ws

'''
problem 4
'''
w0, sigma_w = 20, 5 # a config which does not yeild post synaptic spike
weights_w = sigma_w*np.random.randn(n_out) + w0
# now I need to get n_out number of synapse with above weights
I0 = 100*(10**-12)
tau = 15*(10**-3)
tau_s = tau/4

Sy_list = []
for i in range(n_out):
   Sy = CONST_SYNAPSE(weights_w[i], I0, tau, tau_s)
   Sy_list.append(Sy)

del_tks, del_ws = iterative_synapse_tune(V_pre, Sy_list, -1, 1, max_iter=15)

plt.plot(del_tks, del_ws)
plt.xlabel('mean delta tk in an iteration')
plt.ylabel('mean delta wk in an iteration')
plt.show()
