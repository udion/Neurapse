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










