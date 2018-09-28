THIS CODEBASE IS WRITTEN IN PYTHON.

THE FOLLOWING GENERIC LIBRARIES ARE EXPECTED TO BE PRESENT

numpy
matplotlib
scipy


* The code for this assignment is in src/

* each problem has it's own solution file i.e problem 3 has solution file HW2_3.py  (pther python files are necessary do not mess with them)

* to run a solution to problem, say for problem 4, do

		python HW2_4.py

* Note that problem 3,4,5 require to tune the weights of the synapse population iteratively. Since it doesn't take a large number of iterations (usually <15)
so we have plotted the post synaptic response AFTER EVERY ITERATION, IT WILL POP A GRAPH AFTER EVERY ITERATION, YOU NEED TO CLOSE THE GRAPH TO CONTINUE TUNING

* Problem 3 and 4 asked for graphs of EVERY SYNAPSE UPADATE AT EVERY TRAINING STEPS, there are 100 synapses, it's unreasonable to plot 100 graphs, instead we took the mean over 100 synapses and plotted a single curve representing mean behaviour

* Problem 5 has unnecassrily large number of sub-parts (b and c are really one problem) we have presented the final results after processing b and c, similarly for d.
