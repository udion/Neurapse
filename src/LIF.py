import numpy as np
import matplotlib.pyplot as plt

class LIF():
    def __init__(self, C, gL, V_thresh, El, num_neurons=1):
        self.C = C
        self.gL = gL
        self.V_thresh = V_thresh
        self.El = El
        self.num_neurons = num_neurons
        self.fireflag = [False]*num_neurons
    
    def compute(self, V0, I, delta_t):
        '''
            I : array having current values at all the time [num_neurons X n_t]
            V0 : initial memberane potential [num_neurons, ]
        '''
        self.delta_t = delta_t
        n_t = I.shape[1]
        V = []
        Vi = V0
        print(Vi.shape)
        V.append(Vi)
        for i in range(n_t):
            Vi = self.update_fn(Vi, I[:,i].reshape(self.num_neurons, 1))
            V.append(Vi)
        V = np.concatenate(V, axis=1)
        return V
        
    def update_fn(self, Vi, Ii):
        V_i1 = Vi + ((-1*self.gL*Vi/self.C) + self.gL*self.El/self.C + Ii/self.C)*(self.delta_t - (self.gL*self.delta_t**2)/(2*self.C))

        for idx, f in enumerate(self.fireflag):
            if f == True: #overide the calculation with the reset potential
                self.fireflag[idx] = False
                V_i1[idx] = self.El
        
        for idx, v in enumerate(V_i1):
            if v[0] >= self.V_thresh: #fire
                print(idx, 'firing!!')
                V_i1[idx] = 10*self.V_thresh
                self.fireflag[idx] = True
        
        return V_i1

C=300*(10**-12)
gL=30*(10**-9)
V_thresh=20*(10**-3)
El=-70*(10**-3)
delta_t = 0.1*(10**-3)

Ic0 = np.array([gL*(V_thresh-El)]*int((500*(10**-3)/delta_t)))
Ic1 = np.array([gL*(V_thresh-El+0.0001)]*int((500*(10**-3)/delta_t)))
Ic2 = np.array([gL*(V_thresh-El+0.0002)]*int((500*(10**-3)/delta_t)))
Ic3 = np.array([gL*(V_thresh-El+0.0003)]*int((500*(10**-3)/delta_t)))
Ic0 = Ic0.reshape(1,-1)
Ic1 = Ic1.reshape(1,-1)
Ic2 = Ic2.reshape(1,-1)
Ic3 = Ic3.reshape(1,-1)
I = np.array([Ic0, Ic1, Ic2, Ic3])
I = I.reshape(4, -1)
print(I.shape)

neuron0 = LIF(C, gL, V_thresh, El)
V0 = np.ones(shape=(1,1))
V0 = El*V0
V = neuron0.compute(V0, Ic3, delta_t)
plt.plot(V[0,:], 'r', label='Ic={:.4f} X e-9'.format(Ic3[0,0]*10**9))
plt.ylabel('potential')
plt.xlabel('time')
plt.legend(loc=1)
plt.show()

neuron1 = LIF(C, gL, V_thresh, El, num_neurons=4)
V0 = np.ones(shape=(4,1))
V0 = El*V0
V = neuron1.compute(V0, I, delta_t)
plt.plot(V[0,:], 'r', label='Ic={:.4f} X e-9'.format(I[0,0]*10**9))
plt.plot(V[1,:], 'b', label='Ic={:.4f} X e-9'.format(I[1,0]*10**9))
plt.plot(V[2,:], 'g', label='Ic={:.4f} X e-9'.format(I[2,0]*10**9))
plt.plot(V[3,:], 'y', label='Ic={:.4f} X e-9'.format(I[3,0]*10**9))
plt.ylabel('potential')
plt.xlabel('time')
plt.legend(loc=1)
plt.show()

