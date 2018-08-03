import numpy as np
import matplotlib.pyplot as plt

class IZHIKEVICH():
    def __init__(self, C, Kz, Er, Et, a, b, c, d, V_thresh, num_neurons=1):
        self.C = C
        self.Kz = Kz
        self.Er = Er
        self.Et = Et
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.V_thresh = V_thresh
        self.num_neurons = num_neurons
        self.fireflag = [False]*num_neurons
    
    def compute(self, V0, U0, I, delta_t):
        '''
            I : array having current values at all the time [num_neurons X n_t]
            V0 : initial memberane potential [num_neurons, ]
            U0 : initial U [num_neurons, ]
        '''
        self.delta_t = delta_t
        n_t = I.shape[1]
        V = []
        U = []
        Vi = V0
        Ui = U0
        V.append(Vi)
        U.append(Ui)
        for i in range(n_t):
            Vi, Ui = self.update_fn(Vi, Ui, I[:,i].reshape(self.num_neurons,1))
            V.append(Vi)
            U.append(Ui)
        V = np.concatenate(V, axis=1)
        U = np.concatenate(U, axis=1)
        return V, U
    
    def update_fn(self, Vi, Ui, Ii):
        V_i1 = Vi + self.delta_t*( (1/self.C)*( (self.Kz*(Vi - self.Er)*(Vi - self.Et)) - Ui + Ii ) )
        U_i1 = Ui + self.delta_t*( self.a*(self.b*(Vi - self.Er) - Ui))

        for idx, f in enumerate(self.fireflag):
            if f == True: #overide the calculation with the reset potential
                self.fireflag[idx] = False
                reset_V = self.c
                reset_U = Ui[idx] + self.d
                V_i1[idx,0] = reset_V
                U_i1[idx,0] = reset_U
        
        for idx, v in enumerate(V_i1):
            if v[0] >= self.V_thresh: #fire
                print(idx, 'firing!!')
                V_i1[idx] = 3*self.V_thresh
                self.fireflag[idx] = True
        
        return V_i1, U_i1

######## RS model ############
C1 = 100*(10**-12)
Kz1 = 0.7*(10**-6)
Er1 = -60*(10**-3)
Et1 = -40*(10**-3)
a1 = 0.03*(10**3)
b1 = -2*(10**-9)
c1 = -50*(10**-3)
d1 = 100*(10**-12)
V_thresh1 = 35*(10**-3)
######## IB model ############
C2 = 150*(10**-12)
Kz2 = 1.2*(10**-6)
Er2 = -75*(10**-3)
Et2 = -45*(10**-3)
a2 = 0.01*(10**3)
b2 = 5*(10**-9)
c2 = -56*(10**-3)
d2 = 130*(10**-12)
V_thresh2 = 50*(10**-3)
######## CH model ############
C3 = 50*(10**-12)
Kz3 = 1.5*(10**-6)
Er3 = -60*(10**-3)
Et3 = -40*(10**-3)
a3 = 0.03*(10**3)
b3 = 1*(10**-9)
c3 = -40*(10**-3)
d3 = 150*(10**-12)
V_thresh3 = 25*(10**-3)


neuronRHs = IZHIKEVICH(C1, Kz1, Er1, Et1, a1, b1, c1, d1, V_thresh1, num_neurons=3)
neuronIBs = IZHIKEVICH(C2, Kz2, Er2, Et2, a2, b2, c2, d2, V_thresh2, num_neurons=3)
neuronCHs = IZHIKEVICH(C3, Kz3, Er3, Et3, a3, b3, c3, d3, V_thresh3, num_neurons=3)

delta_t = 0.1*(10**-3)
T = 500*(10**-3)
I1 = np.array([600*(10**-12)]*int(T/delta_t))
I1 = I1.reshape(1,-1)
I2 = np.array([500*(10**-12)]*int(T/delta_t))
I2 = I2.reshape(1,-1)
I3 = np.array([400*(10**-12)]*int(T/delta_t))
I3 = I3.reshape(1,-1)
I = np.concatenate([I1, I2, I3], axis=0)
print(I.shape)

print('I = {:.2f}pA'.format(I1[0,0]*(10**12)))
print('I = {:.2f}pA'.format(I2[0,0]*(10**12)))
print('I = {:.2f}pA'.format(I3[0,0]*(10**12)))

V10 = b1/Kz1 + Et1
U10 = b1*(b1/Kz1 + Et1 - Er1)
#
V20 = b2/Kz2 + Et2
U20 = b2*(b2/Kz2 + Et2 - Er2)
#
V30 = b3/Kz3 + Et3
U30 = b3*(b3/Kz3 + Et3 - Er3)

def simulate_neuron(type):
    if type == 'RH':
        V0, U0 = V10*np.ones(shape=(3,1)), U10*np.ones(shape=(3,1))
        neurons = neuronRHs
    elif type == 'IB':
        V0, U0 = V20*np.ones(shape=(3,1)), U20*np.ones(shape=(3,1))
        neurons = neuronIBs
    elif type == 'CH':
        V0, U0 = V30*np.ones(shape=(3,1)), U30*np.ones(shape=(3,1))
        neurons = neuronCHs
    V, U = neurons.compute(V0, U0, I, delta_t)

    plt.figure(figsize=(15, 20))
    plt.subplot(2,1,1)
    plt.plot(V[0,:], 'r', label='I = {:.2f}pA'.format(I[0,0]*(10**12)))
    plt.plot(V[1,:], 'b', label='I = {:.2f}pA'.format(I[1,0]*(10**12)))
    plt.plot(V[2,:], 'g', label='I = {:.2f}pA'.format(I[2,0]*(10**12)))
    plt.ylabel('membrane potential')
    plt.xlabel('time')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(U[0,:], 'r', label='I = {:.2f}pA'.format(I[0,0]*(10**12)))
    plt.plot(U[1,:], 'b', label='I = {:.2f}pA'.format(I[1,0]*(10**12)))
    plt.plot(U[2,:], 'g', label='I = {:.2f}pA'.format(I[2,0]*(10**12)))
    plt.ylabel('U(t)')
    plt.xlabel('time')
    plt.legend()

    plt.show()

simulate_neuron('RH')
simulate_neuron('IB')
simulate_neuron('CH')



