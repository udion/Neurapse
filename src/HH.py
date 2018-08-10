import numpy as np
import matplotlib.pyplot as plt

class HH():
    def __init__(self, C, E_Na, E_k, E_l, g_Na, g_k, g_l, num_neurons=1):
        self.C = C
        self.E_Na = E_Na
        self.E_k= E_k
        self.E_l = E_l
        self.g_Na = g_Na
        self.g_k = g_k
        self.g_l = g_l
        self.num_neurons = num_neurons
    
    def compute(self, V0, h0, m0, n0, I, delta_t):
        '''
            I : array having current values at all the time [num_neurons X n_t]
            V0 : initial memberane potential [num_neurons, 1]
            m0 : initial m [num_neurons, 1]
            n0 : initial n [num_neurons, 1]
            h0 : initial h [num_neurons, 1]
        '''
        self.delta_t = delta_t
        n_t = I.shape[1]
        V = []
        h = []
        m = []
        n = []
        Vi = V0
        hi = h0
        mi = m0
        ni = n0
        V.append(Vi)
        h.append(hi)
        m.append(mi)
        n.append(ni)
        for i in range(n_t):
            Vi, hi, mi, ni = self.update_fn(Vi, hi, mi, ni, I[:,i].reshape(self.num_neurons,1))
            V.append(Vi)
            h.append(hi)
            m.append(mi)
            n.append(ni)
        V = np.concatenate(V, axis=1)
        h = np.concatenate(h, axis=1)
        m = np.concatenate(m, axis=1)
        n = np.concatenate(n, axis=1)
        return V, h, m, n
    
    def update_fn(self, Vi, hi, mi, ni, Ii):
        I_Na = self.g_Na*(mi**3)*(hi)*(Vi - self.E_Na)
        I_k = self.g_k*(ni**4)*(Vi - self.E_k)
        I_l = self.g_l*(Vi - self.E_l)

        V_i1 = Vi + self.delta_t*( (1/self.C)*( Ii - I_Na - I_k - I_l ))
        # now calculate the h, m, n for the next time step
        h_i1 = hi + self.delta_t*( self.grad_x('h', hi, Vi) )
        m_i1 = mi + self.delta_t*( self.grad_x('m', mi, Vi) )
        n_i1 = ni + self.delta_t*( self.grad_x('n', ni, Vi) )
        return V_i1, h_i1, m_i1, n_i1
    
    def grad_x(self, x, xi, Vi):
        if x == 'h':
            alpha = 0.07*np.exp(-0.05*(Vi*(10**3) + 65)) 
            beta = 1/(1 + np.exp(-0.1*(Vi*(10**3) + 35)))
        elif x == 'm':
            alpha = (0.1*(Vi*(10**3) + 40))/(1 - np.exp(-1*(Vi*(10**3) + 40)/10))
            beta = 4*np.exp(-0.0556*(Vi*(10**3) + 65))
        elif x == 'n':
            alpha = ( 0.01*(Vi*(10**3)+55) )/(1 - np.exp( -1*(Vi*(10**3)+55)/10 ))
            beta = 0.125*np.exp(-1*(Vi*(10**3) + 65)/80)
        grad = alpha*(1-xi) - beta*xi
        return grad
    
C = 1*(10**-6)
E_Na = 50*(10**-3)
E_k = -77*(10**-3)
E_l = -55*(10**-3)
g_Na = 120*(10**-3)
g_k = 36*(10**-3)
g_l = 0.3*(10**-3)
I0 = 15*(10**-6)

T = 30*(10**-3)
delta_t = 0.01*(10**-3)
n_t = int(5*T//delta_t)
I = np.zeros(shape=(1,n_t))
print('I shape : ', I.shape)
for t in range(n_t):
    if t<int(3*T//delta_t) and t>=int(2*T//delta_t):
        I[0,t] = 1
I = I0*I

'''
# to find the initial values for steady state
import numpy as np
from scipy.optimize import newton_krylov

def get_alphabeta(x, Vi):
    if x == 'h':
        alpha = 0.07*np.exp(-0.05*(Vi*(10**3) + 65)) 
        beta = 1/(1 + np.exp(-0.1*(Vi*(10**3) + 35)))
    elif x == 'm':
        alpha = (0.1*(Vi*(10**3) + 40))/(1 - np.exp(-1*(Vi*(10**3) + 40)/10))
        beta = 4*np.exp(-0.0556*(Vi*(10**3) + 65))
    elif x == 'n':
        alpha = ( 0.01*(Vi*(10**3)+55) )/(1 - np.exp( -1*(Vi*(10**3)+55)/10 ))
        beta = 0.125*np.exp(-1*(Vi*(10**3) + 65)/(80))
    
    return alpha/(alpha+beta)

def residual(V):
    r = V - ( ( g_Na*E_Na*(get_alphabeta('m', V)**3)*get_alphabeta('h', V) + g_k*E_k*(get_alphabeta('n', V)**4) + g_l*E_l )/( g_Na*(get_alphabeta('m', V)**3)*get_alphabeta('h', V) + g_k*(get_alphabeta('n', V)**4) + g_l ) )
    return r

guess = np.array([-0.065])
sol = newton_krylov(residual, guess, method='lgmres', verbose=1)
print('Residual: %g' % abs(residual(sol)).max())
print('solution V: {}'.format(sol))
print('solution h: {}'.format(get_alphabeta('h', sol)))
print('solution m: {}'.format(get_alphabeta('m', sol)))
print('solution n: {}'.format(get_alphabeta('n', sol)))

# ------------ output ---------- 
# 0:  |F(x)| = 1.18127e-06; step 1; tol 1.78429e-05
# Residual: 1.18127e-06
# solution V: [-0.06515672]
# solution h: [0.60159082]
# solution m: [0.05196212]
# solution n: [0.31527801]
'''
V0 = -0.06515672*np.ones((1,1))
h0 = 0.60159082*np.ones((1,1))
m0 = 0.05196212*np.ones((1,1))
n0 = 0.31527801*np.ones((1,1))

neuron = HH(C, E_Na, E_k, E_l, g_Na, g_k, g_l)
V, h, m, n = neuron.compute(V0, h0, m0, n0, I, delta_t)
i_Na = g_Na*(m**3)*h*(V-E_Na)
i_k = g_k*(n**4)*(V-E_k)
i_l = g_l*(V-E_l)

P_Na = i_Na*(V-E_Na)
P_k = i_k*(V-E_k)
P_l = i_l*(V-E_l)
P_cv = (-i_Na[:,1:] - i_k[:,1:] - i_l[:,1:] + I)*V[:,1:]

plt.figure(figsize=(10,15))

plt.subplot(3,1,1)
plt.plot(list(range(n_t)), I[0,:])
plt.xlabel('time')
plt.ylabel('current')

plt.subplot(3,1,2)
plt.plot(list(range(n_t)), V[0,1:])
plt.xlabel('time')
plt.ylabel('membrane potential')

plt.subplot(3,1,3)
plt.plot(list(range(n_t)), h[0,1:], 'r', label='h')
plt.plot(list(range(n_t)), m[0,1:], 'g', label='m')
plt.plot(list(range(n_t)), n[0,1:], 'b', label='n')
plt.xlabel('time')
plt.ylabel('parameter values')
plt.legend()
plt.show()

plt.figure(figsize=(10,15))
plt.plot(list(range(n_t)), i_Na[0,1:], 'orange', label='Na')
plt.plot(list(range(n_t)), i_k[0,1:], 'y', label='k')
plt.plot(list(range(n_t)), i_l[0,1:], 'b', label='l')
plt.legend()
plt.xlabel('time')
plt.ylabel('current')
plt.show()

print(P_Na.shape)
print(P_cv.shape)
plt.figure(figsize=(10,15))
plt.plot(list(range(n_t)), P_Na[0,1:], 'orange', label='Na')
plt.plot(list(range(n_t)), P_k[0,1:], 'y', label='k')
plt.plot(list(range(n_t)), P_l[0,1:], 'b', label='l')
plt.plot(list(range(n_t)), P_cv[0,:], 'r', label='capacitor')
plt.legend()
plt.xlabel('time')
plt.ylabel('power')
plt.show()



