import numpy as np
import matplotlib.pyplot as plt
from Synapes import *
from SpikeTrains import *
from AEF import AEF

'''
problem 5
'''
n_out = 100
T = 500*(10**-3)
delta_t = 0.1*(10**-3)
lamb = 1
n_t = int(T/delta_t)

ST1 = POISSON_SPIKE_TRAIN(T=T, delta_t=delta_t, lamb=lamb, n_out=n_out)
ST2 = POISSON_SPIKE_TRAIN(T=T, delta_t=delta_t, lamb=lamb, n_out=n_out)
V_pre1, SI1 = ST1.V_train, ST1.spike_instants
V_pre2, SI2 = ST2.V_train, ST2.spike_instants

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

def simulate_neuron(type, I_total, name=''):
    if type == 'RH':
        V0, U0 = V10*np.ones(shape=(1,1)), U10*np.ones(shape=(1,1))
        neurons = neuronRHs
    else:
        print('not supported here!, check AEF file')
        return
    V, U = neurons.compute(V0, U0, I_total, delta_t)
    fig = plt.figure(figsize=(15, 20))

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
    if name != '':
        fig.savefig(name)
    return V, U

# Now, assuming the initial state is such
# that there are no post synaptic spikes
# then I need to iterate and update weight until
# I get a single post synaptic spike, the following
# snippet will do the same

def iterative_synapse_tune(V_pre, SI, Sy_list, upd_coeff, gamma, max_iter=500):
    '''
    This will update the weights until one spike is seen (if upd_coeff == 1)
    or all spikes die (if upd_coeff == 0) 
    '''
    itr = 0
    while itr<max_iter:
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
                break
        elif upd_coeff == -1:
            if np.sum(V_post[0, :] > 0) <= 1:
                print('post neuron spiked only once at iter {}'.format(itr))
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
            Sy_list[syn_id].weight_update(gamma, syn_delta_t, upd_coeff)

        itr = itr+1
    return Sy_list

w0, sigma_w = 200, 20 # a config which does not yeild post synaptic spike
weights_w = sigma_w*np.random.randn(n_out) + w0
# now I need to get n_out number of synapse with above weights
I0 = 100*(10**-12)
tau = 15*(10**-3)
tau_s = tau/4

Sy_list = []
for i in range(n_out):
   Sy = CONST_SYNAPSE(weights_w[i], I0, tau, tau_s)
   Sy_list.append(Sy)

# seeing the response on the 2 family of spike trains with inital synapse configs
print('seeing the response on the 2 family of spike trains with inital synapse configs ...\n')
I_Sy_list = []
for i in range(n_out):
    I = Sy.getI(V_pre1[i,:].reshape(1,-1), [SI1[i]], delta_t)
    I_Sy_list.append(I)
I_total = 0
for I_t in I_Sy_list:
    I_total = I_t+I_total
simulate_neuron('RH', I_total, 'response_S1_pre.png')

I_Sy_list = []
for i in range(n_out):
    I = Sy.getI(V_pre2[i,:].reshape(1,-1), [SI2[i]], delta_t)
    I_Sy_list.append(I)
I_total = 0
for I_t in I_Sy_list:
    I_total = I_t+I_total
simulate_neuron('RH', I_total, 'response_S2_pre.png')


# fine tuning through iteration
print('tuning so that it does not responds well to S1..\n')
Sy_list = iterative_synapse_tune(V_pre1, SI1, Sy_list, -1, 1, max_iter=4)
###
print('tuning so that it does responds well to S2..\n')
Sy_list = iterative_synapse_tune(V_pre2, SI2, Sy_list, 1, 1, max_iter=4)


### results of final tuned network
print('tuned networks response on S1 ...\n')
I_Sy_list1 = []
for i in range(n_out):
    I = Sy_list[i].getI(V_pre1[i,:].reshape(1,-1), [SI1[i]], delta_t)
    I_Sy_list1.append(I)
I_total1 = 0
for I_t in I_Sy_list1:
    I_total1 = I_t+I_total1
simulate_neuron('RH', I_total1, 'response_S1_post.png')

print('tuned networks response on S2 ...\n')
I_Sy_list2 = []
for i in range(n_out):
    I = Sy_list[i].getI(V_pre2[i,:].reshape(1,-1), [SI2[i]], delta_t)
    I_Sy_list2.append(I)
I_total2 = 0
for I_t in I_Sy_list2:
    I_total2 = I_t+I_total2
simulate_neuron('RH', I_total2, 'response_S2_post.png')

print('weight configs ...')
for z,Sy in enumerate(Sy_list):
    print('synapse {}: w={}'.format(z, Sy.w))

###############################################################
print('now tuning for reverse config ...')

# fine tuning through iteration
print('tuning so that it does not responds well to S2..\n')
Sy_list = iterative_synapse_tune(V_pre2, SI2, Sy_list, -1, 1, max_iter=4)
###
print('tuning so that it does responds well to S1..\n')
Sy_list = iterative_synapse_tune(V_pre1, SI1, Sy_list, 1, 1, max_iter=10)


### results of final tuned network
print('tuned networks response on S1 ...\n')
I_Sy_list1 = []
for i in range(n_out):
    I = Sy_list[i].getI(V_pre1[i,:].reshape(1,-1), [SI1[i]], delta_t)
    I_Sy_list1.append(I)
I_total1 = 0
for I_t in I_Sy_list1:
    I_total1 = I_t+I_total1
simulate_neuron('RH', I_total1, 'response_S1_post_rev.png')

print('tuned networks response on S2 ...\n')
I_Sy_list2 = []
for i in range(n_out):
    I = Sy_list[i].getI(V_pre2[i,:].reshape(1,-1), [SI2[i]], delta_t)
    I_Sy_list2.append(I)
I_total2 = 0
for I_t in I_Sy_list2:
    I_total2 = I_t+I_total2
simulate_neuron('RH', I_total2, 'response_S2_post_rev.png')

print('weight configs ...')
for z,Sy in enumerate(Sy_list):
    print('synapse {}: w={}'.format(z, Sy.w))