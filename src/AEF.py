import numpy as np
import matplotlib.pyplot as plt

e = 2.713

class AEF():
    def __init__(self, C, gl, El, Vt, Delt, a, tw, b, Vr, num_neurons=1):
        self.C = C
        self.gl = gl
        self.El = El
        self.Vt = Vt
        self.Delt = Delt
        self.a = a
        self.tw = tw
        self.b = b
        self.Vr = Vr
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
        V_i1 = Vi + self.delta_t*( (1/self.C)*( -self.gl*(Vi - self.El) + self.gl*self.Delt*(e**((Vi - self.Vt)/self.Delt)) - Ui + Ii ) )
        U_i1 = Ui + self.delta_t*( (1/self.tw)*( self.a*(Vi - self.El) - Ui))

        for idx, f in enumerate(self.fireflag):
            if f == True: #overide the calculation with the reset potential
                self.fireflag[idx] = False
                reset_V = self.Vr
                reset_U = Ui[idx] + self.b
                V_i1[idx,0] = reset_V
                U_i1[idx,0] = reset_U
        
        ## Please check
        for idx, v in enumerate(V_i1):
            if v[0] >= self.Vr: #fire
                print(idx, 'firing!!')
                V_i1[idx] = 3*self.Vr
                self.fireflag[idx] = True
        
        return V_i1, U_i1


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
######## IB model ############
C2 = 130*(10**-12)
gl2 = 18*(10**-9)
El2 = -58*(10**-3)
Vt2 = -50*(10**-3)
Delt2 = 2*(10**-3)
a2 = 4*(10**-9)
tw2 = 150*(10**-3)
b2 = 120*(10**-12)
Vr2 = -50*(10**-3)
######## CH model ############
C3 = 200*(10**-12)
gl3 = 10*(10**-9)
El3 = -58*(10**-3)
Vt3 = -50*(10**-3)
Delt3 = 2*(10**-3)
a3 = 2*(10**-9)
tw3 = 120*(10**-3)
b3 = 100*(10**-12)
Vr3 = -46*(10**-3)


neuronRHs = AEF(C1, gl1, El1, Vt1, Delt1, a1, tw1, b1, Vr1, num_neurons=3)
neuronIBs = AEF(C1, gl2, El2, Vt2, Delt2, a2, tw2, b2, Vr2, num_neurons=3)
neuronCHs = AEF(C1, gl3, El3, Vt3, Delt3, a3, tw3, b3, Vr3, num_neurons=3)

delta_t = 0.1*(10**-3)
T = 500*(10**-3)
I1 = np.array([250*(10**-12)]*int(T/delta_t))
I1 = I1.reshape(1,-1)
I2 = np.array([350*(10**-12)]*int(T/delta_t))
I2 = I2.reshape(1,-1)
I3 = np.array([450*(10**-12)]*int(T/delta_t))
I3 = I3.reshape(1,-1)
I = np.concatenate([I1, I2, I3], axis=0)
print(I.shape)

print('I = {:.2f}pA'.format(I1[0,0]*(10**12)))
print('I = {:.2f}pA'.format(I2[0,0]*(10**12)))
print('I = {:.2f}pA'.format(I3[0,0]*(10**12)))

V0 = np.linspace(-10000**-3, 10000**-3, 10000)
y1 = (1 + a1/gl1)*(V0 - El1)/(gl1*Delt1)
y2 = e**((V0 - Vt1)/Delt1)
# Find intersection of y1, y2

# Steady state values of V and U for I app = 0
V10 = []
U10 = []
#
V20 = []
U20 = []
#
V30 = []
U30 = []

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



