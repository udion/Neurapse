# Neurapse

## 3rd Gen Neural Networks
Current Deep Learning methods (2nd Gen) set very impressive state of the art results, although the building blocks such as *convolutions etc* are biologically inspired ***they still are not as efficient as computations happening at many biological neural networks***

Spiking Neural Networks (SNNs) is an attempt to simulate biological networks closely. Broadly the framework consists of **Spikes**, **Neurons**, **Synapses**, **Networks**.
**Neurapse** is a package in python which implements some of the fundamental blocks of SNN and is written in a manner so that it can easily be extended and customized.

* Neurons : Hodgkin Huxley (HH), Adaptive Exponential Fire (AEF) , Leaky integrate and Fire (LIF), IZHIKEVICH

* Synapses: Constant Synapse (No STDP), Plastic Synapses (2 kinds of STDP)

* Networks: Feed Forward using LIF, Dynamic Random Networks

## How to use?
clone or fork this repository by `git clone https://github.com/udion/Neurapse`. Make sure you have the dependencies given in the `requirements.txt` (*So far it only requires numpy, matplotlib*)

Some examples are given with the comments in `examples*.py`. 

### Importing a neuron
```python
import numpy as np
import matplotlib.pyplot as plt

import Neurapse.Neurons as HH #importing Hodgkin-Huxley neuron
import Neurapse.utils.CURRENTS as Cur #to generate currents (Square pulse in this example)
```

Neurons in SNN frameworks are described using certain parameters such as *Capacitance, Resting potentials, Time constants **etc***

Hodgkin-Huxley neuron in particular has the following parameters :
```python
C = 1e-6
E_Na = 50e-3
E_k = -77e-3
E_l = -55e-3
g_Na = 120e-3
g_k = 36e-3
g_l = 0.3e-3
I0 = 15e-6

T = 30e-3 # Time in seconds
delta_t = 1e-5 # quanta in which time updates in seconds
n_t = int(5*T//delta_t)+1 # Total time of simulation is 5*T, hence number of time-steps is n_t
```
Let's generate the input current and visualise it
```python
Sq1 = Cur.SQUARE_PULSE(t_start=6000, t_end=9000, T=n_t)
I = Sq1.generate() # normalised current i.e maximum amplitude is 1 unit, of shape [1 X T]
I = I0*I # I is input current to the neuron in micro-Ampere 

plt.plot(I)
plt.xlabel('time')
plt.ylabel('current')
```










