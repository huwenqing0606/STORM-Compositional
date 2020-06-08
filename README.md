<b>STOchastic Recursive Momentum for Compositional Optimization</b>

an article by Huizhuo Yuan and Wenqing Hu

Code for STORM-Compositional Optimization

<i>Authors</i>: Jiaojiao Yang (Anhui Normal University, Wuhu, Anhui, P.R.China) for topics (1) and (2) and Wenqing Hu (Missouri S&T) for topic (3)

<i>Remark</i>: The part of the code for SARAH-C and other previously announced algorithms are adapted from the orginal code for SARAH-Compositional open sourced at https://github.com/angeoz/SCGD, and we try to remain the parts for these previously announced algorithms as much intact as possible. However, the part of the code for our STORM-C in our paper are completely newly developed. This is in order to set a standard to compare the convergence properties between our new STORM-Compositional with the original SARAH-Compositional and other previously announced benchmark algorithms. In a few places when we find inconsistencies of the original code with our calculations, that cause significant inaccuracies in experimental performance, we modified the original code correpondingly. 

(1) Folder "portfolio"

Applying STORM-Compositional to portfolio optimization. (a) folder "data", the data set; (b) run_pot2.m, the run file; (c) opt_VR.m, the optimization algorithms including SARAH-Compositional and STORM-Compositional; (d) compute_min_val.m, calculate the minimal objective function; (e) computer_port.m, calculate the objective function and its gradients.

(2) Folder "rl"

Applying STORM-Compositional to value-function evaluation in reinforcement learning. (a) run_cov2.m, the run file; (b) opt_RL.m, the optimization algorithms including SARAH-Compositional and STORM-Compositional; (c) compute_obj.m, calculate the objective function and its gradients.

(3) Folder "SNE"

Applying STORM-Compositional to Stochastic Neighbor Embedding. (a) run_tsne.m, the run file; (b) opt_TSNE.m, the optimization algorithms inclucing SARAH-Compositional and STORM-Compositional; (c) compute_tsne.m, calculate the objective function and its gradients; (d) loadMNISTLabels.m, loadMNISTImages.m, t10k-images-idx3-ubyte, data source and loading files for MNIST images set.
