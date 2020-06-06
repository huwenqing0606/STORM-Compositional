<b>STOchastic Recursive Momentum for Compositional Optimization</b>

Code for STORM-Compositional Optimization

Remark: Part of the Code are directly adapted from the orginal code for SARAH-Compositional open sourced at https://github.com/angeoz/SCGD. This is in order to compare the convergence properties between our new STORM-Compositional with the original SARAH-Compositional and other benchmark algorithms. Only in a few places, we modified the original code for SARAH-Compositional available at https://github.com/angeoz/SCGD as we find some parts of this code inconsistent with the announced algorithm SARAH-Compositional or the calculations in the model problem described in that paper (available at https://arxiv.org/pdf/1912.13515.pdf).

(1) Folder "portfolio"

Applying STORM-Compositional to portfolio optimization. (a) folder "data", the data set; (b) run_pot2.m, the run file; (c) opt_VR.m, the optimization algorithms including SARAH-Compositional and STORM-Compositional; (d) compute_min_val.m, calculate the minimal objective function; (e) computer_port.m, calculate the objective function and its gradients.

(2) Folder "rl"

Applying STORM-Compositional to value-function evaluation in reinforcement learning. (a) run_cov2.m, the run file; (b) opt_RL.m, the optimization algorithms including SARAH-Compositional and STORM-Compositional; (c) compute_obj.m, calculate the objective function and its gradients.
