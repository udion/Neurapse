import numpy as np
import matplotlib.pyplot as plt

class LIF():
    def __init__(self, C, gL, V_thresh, El):
        self.C = C
        self.gL = gL
        self.V_thresh = V_thresh
        self.El = El
        self.fireflag = False
    
    def compute(self, V0, I, delta_t):
        '''
            I : array having current values at all the time
            V0 : initial memberane potential
        '''
        self.delta_t = delta_t
        n_t = I.shape[1] #I is 1Xnt
        V = []
        Vi = V0
        V.append(Vi)
        for i in range(n_t):
            Vi = self.update_fn(Vi, I[0,i])
            V.append(Vi)
        return V
        
    def update_fn(self, Vi, Ii):
        if self.fireflag == True:
            self.fireflag = False
            return self.El
        else:
            V_i1 = Vi + ((-1*self.gL*Vi/self.C) + self.gL*self.El/self.C + Ii/self.C)*(self.delta_t - (self.gL*self.delta_t**2)/(2*self.C))
            if V_i1 >= self.V_thresh: #fire
                print('firing!!')
                V_i1 = 10*self.V_thresh
                self.fireflag = True
            return V_i1

C=300*(10**-12)
gL=30*(10**-9)
V_thresh=20*(10**-3)
El=-70*(10**-3)
delta_t = 0.1*(10**-3)

Ic = np.array([gL*(V_thresh-El)]*int((500*(10**-3)/delta_t)))
Ic1 = np.array([gL*(V_thresh-El+0.0001)]*int((500*(10**-3)/delta_t)))
Ic2 = np.array([gL*(V_thresh-El+0.0002)]*int((500*(10**-3)/delta_t)))
Ic3 = np.array([gL*(V_thresh-El+0.0003)]*int((500*(10**-3)/delta_t)))
Ic = Ic.reshape(1,-1)
Ic1 = Ic1.reshape(1,-1)
Ic2 = Ic2.reshape(1,-1)
Ic3 = Ic3.reshape(1,-1)
# print(Ic.shape)

neuron1 = LIF(C, gL, V_thresh, El)

V = neuron1.compute(El, Ic, delta_t)
V1 = neuron1.compute(El, Ic1, delta_t)
V2 = neuron1.compute(El, Ic2, delta_t)
V3 = neuron1.compute(El, Ic3, delta_t)

plt.plot(V, 'r', label='Ic={:.4f} X e-9'.format(Ic[0,0]*10**9))
plt.plot(V1, 'b', label='Ic={:.4f} X e-9'.format(Ic1[0,0]*10**9))
plt.plot(V2, 'g', label='Ic={:.4f} X e-9'.format(Ic2[0,0]*10**9))
plt.plot(V3, 'y', label='Ic={:.4f} X e-9'.format(Ic3[0,0]*10**9))
plt.ylabel('potential')
plt.xlabel('time')
plt.legend(loc=1)
plt.show()

