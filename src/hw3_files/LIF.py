import numpy as np
import matplotlib.pyplot as plt
from CURRENTS import *

class LIF():
    def __init__(self, C, gL, V_thresh, El, Rp, num_neurons=1):
        self.C = C
        self.gL = gL
        self.V_thresh = V_thresh
        self.El = El
        self.Rp = Rp
        self.num_neurons = num_neurons
        self.fireflag = [False]*num_neurons
        self.refract = [None]*num_neurons
        self.I_synt = np.zeros(shape=(num_neurons,1))
        self.ins_t = 0
    
    def inject(self, t, I_synt):
        self.ins_t = t
        self.I_synt = I_synt
    
    def compute(self, V0, I, delta_t):
        '''
            I : array having current values at all the time [num_neurons X n_t]
            V0 : initial memberane potential [num_neurons, ]
        '''
        self.delta_t = delta_t
        self.n_t = I.shape[1]
        self.n_Rp = int(self.Rp/self.delta_t)
        print('n_rp ', self.n_Rp)
        # return
        V = []
        Vi = V0
        print(Vi.shape)
        V.append(Vi)
        for i in range(self.n_t):
            Vi = self.update_fn(Vi, I[:,i].reshape(self.num_neurons, 1)+self.I_synt, self.delta_t)
            V.append(Vi)
        V = np.concatenate(V, axis=1)
        return V
        
    def update_fn(self, Vi, Ii, delta_t):
        self.delta_t = delta_t
        self.n_Rp = int(self.Rp/self.delta_t)
        V_i1 = Vi + ((-1*self.gL*Vi/self.C) + self.gL*self.El/self.C + Ii/self.C)*(self.delta_t - (self.gL*self.delta_t**2)/(2*self.C))

        for idx, f in enumerate(self.fireflag):
            if f == True: #overide the calculation with the reset potential
                self.fireflag[idx] = False
                V_i1[idx] = self.El
                self.refract[idx] = self.n_Rp
        # print(self.refract)
        
        for idx, r in enumerate(self.refract):
            if r != None:
                V_i1[idx] = self.El
                if self.refract[idx]-1 <= 0:
                    self.refract[idx] = None
                else:
                    self.refract[idx] = self.refract[idx]-1
        
        for idx, v in enumerate(V_i1):
            if v[0] >= self.V_thresh: #fire
                # print(idx, 'firing!!')
                V_i1[idx] = 10*self.V_thresh
                self.fireflag[idx] = True
        
        return V_i1
'''
usage :

C=300*(10**-12)
gL=30*(10**-9)
V_thresh=20*(10**-3)
El=-70*(10**-3)
delta_t = 0.1*(10**-3)
T = 500*(10**-3)
Rp = 2*(10**-3)
# Rp = 0 

# Ic0 = np.array([gL*(V_thresh-El)]*int((500*(10**-3)/delta_t)))
# Ic1 = np.array([gL*(V_thresh-El+0.0001)]*int((500*(10**-3)/delta_t)))
# Ic2 = np.array([gL*(V_thresh-El+0.0002)]*int((500*(10**-3)/delta_t)))
# Ic3 = np.array([gL*(V_thresh-El+0.0003)]*int((500*(10**-3)/delta_t)))
# Ic0 = Ic0.reshape(1,-1)
# Ic1 = Ic1.reshape(1,-1)
# Ic2 = Ic2.reshape(1,-1)
# Ic3 = Ic3.reshape(1,-1)
# I = np.array([Ic0, Ic1, Ic2, Ic3])
# I = I.reshape(4, -1)

num_neurons = 10
I = np.ones(shape=(num_neurons, int(T//delta_t)))
for i in range(num_neurons):
    I[i,:] = (1+i*0.1)*gL*(V_thresh-El)*I[i,:]
print(I.shape)

neurons = LIF(C, gL, V_thresh, El, Rp, num_neurons=num_neurons)
V0 = np.ones(shape=(10,1))
V0 = El*V0
V = neurons.compute(V0, I, delta_t)

plt.figure(figsize=(175,135))
plt.subplot(2,2,1)
plt.plot(V[1,:], 'r', label='I={:.4f} X e-9'.format(I[1,0]*10**9))
plt.ylabel('potential')
plt.xlabel('time')
plt.legend(loc=1)
plt.title('neuron 2')

plt.subplot(2,2,2)
plt.plot(V[3,:], 'r', label='I={:.4f} X e-9'.format(I[3,0]*10**9))
plt.ylabel('potential')
plt.xlabel('time')
plt.legend(loc=1)
plt.title('neuron 4')

plt.subplot(2,2,3)
plt.plot(V[5,:], 'r', label='I={:.4f} X e-9'.format(I[5,0]*10**9))
plt.ylabel('potential')
plt.xlabel('time')
plt.legend(loc=1)
plt.title('neuron 6')

plt.subplot(2,2,4)
plt.plot(V[7,:], 'r', label='I={:.4f} X e-9'.format(I[7,0]*10**9))
plt.ylabel('potential')
plt.xlabel('time')
plt.legend(loc=1)
plt.title('neuron 7')
plt.show()

print('V shape :', V.shape)
print('plotting the variation of time to spike with input current magnitude')
tt_spike = []
applied_I = []
for i in range(num_neurons):
    tt_spike.append(np.argmax(V[i,:])*delta_t)
    applied_I.append(I[i,0])
tt_spike = np.array(tt_spike)
applied_I = np.array(applied_I)

plt.plot(applied_I, tt_spike)
plt.xlabel('applied current')
plt.ylabel('time interval between spikes')
plt.show()
'''

'''
usage: with sqare pulse 

C=300*(10**-12)
gL=30*(10**-9)
V_thresh=20*(10**-3)
El=-70*(10**-3)
delta_t = 0.1*(10**-3)
T = 500*(10**-3)
Rp = 2*(10**-3)
# Rp = 0 

Ic0 = np.array([gL*(V_thresh-El)]*int((500*(10**-3)/delta_t)))
num_neurons = 1
I = np.ones(shape=(num_neurons, int(T//delta_t)))
for i in range(num_neurons):
    I[i,:] = 3*gL*(V_thresh-El+0.0003)*SQUARE_PULSE(0, 100, int(T//delta_t)).generate()
print(I.shape)

neurons = LIF(C, gL, V_thresh, El, Rp, num_neurons=num_neurons)
V0 = np.ones(shape=(1,1))
V0 = El*V0
V = neurons.compute(V0, I, delta_t)

plt.figure(figsize=(175,135))
plt.subplot(2,2,1)
plt.plot(V[0,:], 'r')
plt.ylabel('potential')
plt.xlabel('time')
plt.legend(loc=1)
plt.title('neuron 2')
plt.show()
'''