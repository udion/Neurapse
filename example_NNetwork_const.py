import numpy as np
import matplotlib.pyplot as plt
import Neurapse.utils.CURRENTS as Cur
from Neurapse.Networks import NNetwork_Const

Fanout = [[0,1],[0,1],[0,1]]
W = [[3000,3000],[3000,3000],[3000,3000]]
Tau = [[1e-3,8e-3],[5e-3,5e-3],[9e-3,1e-3]]  

Net = NNetwork_Const(Fanout, W, Tau, 3, 2)

I_pre = np.array([
    50e-9*Cur.SQUARE_PULSE(0, 10, 10000).generate(),
    50e-9*Cur.SQUARE_PULSE(40, 50, 10000).generate(),
    50e-9*Cur.SQUARE_PULSE(80, 90, 10000).generate(),
]).reshape(3,-1)

print(I_pre.shape)
V_pre_response, V_post_response, I_sy_list, I_post = Net.compute(I_pre, 1e-4)

for i in range(2):
    plt.plot(V_post_response[i,:], label='post-{}'.format(i))
plt.legend()
plt.xlabel('time')
plt.ylabel('V')
plt.title('response of the post neurons')
plt.tight_layout()
plt.show()