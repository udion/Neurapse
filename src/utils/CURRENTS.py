import numpy as np
import random

class SQUARE_PULSE():
    def __init__(self, t_start, t_end, T):
        self.t_start = t_start
        self.t_end = t_end
        self.T = T
        self.current = np.array([0]*(T)).reshape(1,-1)
        self.generate()

    def generate(self):
        for t in range(self.t_start, self.t_end+1):
            self.current[0, t] = 1
        return self.current
'''
import matplotlib.pyplot as plt
SQ = SQUARE_PULSE(10, 50, 200).generate()
plt.plot(SQ[0,:])
plt.show()
'''


