import numpy as np

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
            V0 : initial memberane potential [num_neurons, 1]
            U0 : initial U [num_neurons, 1]
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
        V_i1 = Vi + self.delta_t*( (1/self.C)*( -self.gl*(Vi - self.El) + self.gl*self.Delt*(np.exp((Vi - self.Vt)/self.Delt)) - Ui + Ii ) )
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
            if v[0] >= 0: #fire
                print(idx, 'firing!!')
                V_i1[idx] = 2e-3
                self.fireflag[idx] = True
        
        return V_i1, U_i1

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