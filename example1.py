import matplotlib.pyplot as plt
import numpy as np

# import neurons.HH as HH
import Neurapse.Neurons as HH
import Neurapse.utils.CURRENTS as Cur
from Neurapse.Networks import NNetwork_Const

C = 1e-6
E_Na = 50e-3
E_k = -77e-3
E_l = -55e-3
g_Na = 120e-3
g_k = 36e-3
g_l = 0.3e-3
I0 = 15e-6
T = 30e-3
delta_t = 1e-5
n_t = int(5*T//delta_t)+1 #otherwise one-timestep is gone

Sq1 = Cur.SQUARE_PULSE(t_start=6000, t_end=9000, T=n_t)
I = Sq1.generate()
I = I0*I
print(I.shape)
plt.plot(I[0,:])
plt.xlabel('time')
plt.ylabel('applied current')
plt.tight_layout()
plt.show()


N = HH.HH(C, E_Na, E_k, E_l, g_Na, g_k, g_l)
print(N)

V0 = -0.06515672*np.ones((1,1))
h0 = 0.60159082*np.ones((1,1))
m0 = 0.05196212*np.ones((1,1))
n0 = 0.31527801*np.ones((1,1))

V, h, m, n = N.compute(V0, h0, m0, n0, I, delta_t)
i_Na = g_Na*(m**3)*h*(V-E_Na)
i_k = g_k*(n**4)*(V-E_k)
i_l = g_l*(V-E_l)

plt.figure(figsize=(8,3))
plt.subplot(1,2,1)
plt.plot(V[0,1:])
plt.xlabel('time')
plt.ylabel('membrane potential')

# plt.subplot(2,2,2)
# plt.plot(list(range(n_t)), h[0,1:], 'r', label='h')
# plt.plot(list(range(n_t)), m[0,1:], 'g', label='m')
# plt.plot(list(range(n_t)), n[0,1:], 'b', label='n')
# plt.xlabel('time')
# plt.ylabel('parameter values')
# plt.legend()
# plt.show()

plt.subplot(1,2,2)
plt.plot(i_Na[0,:], 'orange', label='Na')
plt.plot(i_k[0,:], 'y', label='k')
plt.plot(i_l[0,:], 'b', label='l')
plt.legend()
plt.xlabel('time')
plt.ylabel('channel current')
plt.tight_layout()
plt.show()

# Fanout = [
#     [0,1],
#     [0,1],
#     [0,1]
# ]
# W = [
#     [3000,3000],
#     [3000,3000],
#     [3000,3000]
# ]
# Tau = [
#     [1e-3,8e-3],
#     [5e-3,5e-3],
#     [9e-3,1e-3]
# ]  

# A = NNetwork_Const(Fanout, W, Tau, 3, 2)
# print(A)

# I_pre = np.array([
#     50e-9*Cur.SQUARE_PULSE(0, 10, 10000).generate(),
#     50e-9*Cur.SQUARE_PULSE(40, 50, 10000).generate(),
#     50e-9*Cur.SQUARE_PULSE(80, 90, 10000).generate(),
# ]).reshape(3,-1)

# print(I_pre.shape)
# A.compute(I_pre, 1e-4)
# A.display(1)






