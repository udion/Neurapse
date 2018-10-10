import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def ode_solver(update_fn, V0, I_ext):
    num_nurons = V0.shape[0]
    Vi = V0
    V = [Vi]
    n_t = I_ext.shape[1]
    for i in range(n_t):
        Vi = update_fn(Vi, I_ext[:,i])
        V.append(Vi)
        print(i, Vi)
    return np.array(V)

def LIF_update(Vi, Ii):
    C = 300*(10**-12)
    gL = 30*(10**-9)
    VT = 20*(10**-3)
    EL = -70*(10**-3)
    delta_t = 0.1*(10**-3)

    Vi_1 = Vi + ((-1*gL*Vi/C) + gL*EL/C + Ii/C)*(delta_t - (gL*delta_t**2)/(2*C))
    return Vi_1

C = 300*(10**-12)
gL = 30*(10**-9)
VT = 20*(10**-3)
EL = -70*(10**-3)
delta_t = 0.1*(10**-3)
Ic = np.array([gL*(VT-EL)]*int((500*(10**-3)/delta_t)))
V0 = EL
num_nurons = 10
V0 = np.array([V0]*num_nurons)
Vi = V0
I = []
for k in range(num_nurons):
    I.append((1+k*0.1)*Ic)
I = np.array(I)
print(I.shape)

V = ode_solver(LIF_update, V0, I)
for k in range(num_nurons):
    plt.plot(V[:,k])
plt.show()
