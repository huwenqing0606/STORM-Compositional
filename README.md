<b>STOchastic Recursive Momentum for Compositional Optimization</b>

Code for STORM-Compositional Optimization

Author: Jiaojiao Yang (Anhui Normal University, Wuhu, Anhui, P.R.China)

<i>Remark</i>: The part of the code for SARAH-C and other previously announced algorithms are directly adapted from the orginal code for SARAH-Compositional open sourced at https://github.com/angeoz/SCGD, and we try to remain them as intact as possible. Based on this, we developed the code for STORM-C in our paper. This is in order to compare the convergence properties between our new STORM-Compositional with the original SARAH-Compositional and other benchmark algorithms. 

(1) Folder "portfolio"

Applying STORM-Compositional to portfolio optimization. (a) folder "data", the data set; (b) run_pot2.m, the run file; (c) opt_VR.m, the optimization algorithms including SARAH-Compositional and STORM-Compositional; (d) compute_min_val.m, calculate the minimal objective function; (e) computer_port.m, calculate the objective function and its gradients.

(2) Folder "rl"

Applying STORM-Compositional to value-function evaluation in reinforcement learning. (a) run_cov2.m, the run file; (b) opt_RL.m, the optimization algorithms including SARAH-Compositional and STORM-Compositional; (c) compute_obj.m, calculate the objective function and its gradients.
