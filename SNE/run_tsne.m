%%Runfile For STORM-Compositional over stochastic neighbor embedding

%code between separating dashed lines are for STORM, and are newly developed by us
%code else where are adapted from SARAH-Compositional and not changed (see http://github.com/angeoz/SCGD) to make a comparison
%we try to keep the original SARAH-Compositional code as much intact as we can

%author: Jiaojiao Yang (Anhui Normal University)

clearvars;

rng(1, 'twister');

config.l1 = 0;
%config.kappa = 20;

%mu = ones(1, 200)*4;
%A = randn(200, 200);
%[u, s, v] = svd(A);
%s = eye(200);
%s(200, 200) = config.kappa;
%sigma = u*s*u';
%data = mvnrnd(mu, sigma, 2000);
%save('data_cov_4.mat', 'data');
%n = 400;
%d = 100;
%P = unifrnd(0, 1, [n, n]);
%P = P + 1e-5;
%P = P ./ sum(P, 2);
%R = unifrnd(0, 1, [n, n]);
%F = unifrnd(0, 1, [n, d]);
%data.P = P;
%data.R = R;
%data.F = F;
%load data_cov_20;

n = 2000;
m = 784;
d = 10;
dataO = loadMNISTImages('t10k-images-idx3-ubyte');
dataO = dataO(:,1:n)'/10;
dataI = Dist(dataO, 0);
data.P = dataI ./ (sum(dataI, 1)-1);
config.m = d;
config.mm = 1;

di = 1;

%rng(1);
%minval = compute_min_val(data, config);
minval = 1.5221e4;
config.gamma = 0.95;
config.max_iters = 10;  
config.max_epochs = 10;
config.outer_bs = 1000; 
config.inner_bs = 5; 
config.lr = 1e-5;


config.beta = 0.9;
config.opt = 1;
[svrg, grad_svrg, norm_svrg] = opt_TSNE(data, config);
grad_svrg = grad_svrg/n;
config.opt = 2;
config.dec = 1;
[spider, grad_spider, norm_spider] = opt_TSNE(data, config);
grad_spider = grad_spider/n;
config.dec = 0;
config.opt = 0;
[scgd, grad_scgd, norm_scgd] = opt_TSNE(data, config);
grad_scgd = grad_scgd/n;
config.opt = 3;
[ascpg, grad_ascpg, norm_ascpg] = opt_TSNE(data, config);
grad_ascpg = grad_ascpg/n;

minval = min(min(svrg), min(spider));


%STORM-BEGIN-------------------------------------------------------------------------------------------------------------------------------
   
%set do or not do normalization step in STORM
config.STORM_ifnormalization = 1;
%set with or without replacement in minibatch sampling in STORM, with replacement = 1
config.STORM_ifreplace = 1;
%set the STORM single loop batchsizes, learning rate and the a parameters
config.STORM_eps = 0.1;
config.STORM_max_inner_iters = 7; 
%STORM only uses one loop, but we decompose it into epochs and each epoch has same iteration as other compositional optimization algorithms
%if the batchsizes of STORM are small and cannot match the IFO queries of other Compositional algorithms, we tune this inner iteration to be larger correspondingly
config.STORM_lr = 0.1;
    
config.STORM_initial_bs=100;
config.STORM_loop_bs_g=100;
config.STORM_loop_bs_G=100;
config.STORM_loop_bs_F=100;

config.STORM_a_g=0.01;
config.STORM_a_G=0.01;
config.STORM_a_F=0.01;
    
%the option of using STORM
config.opt = 4;
config.max_epochs = 10; %one can tune the STORM max epochs to reach similar IFO compared with SARAH-C
[storm, grad_storm, norm_storm] = opt_TSNE(data, config);
grad_storm = grad_storm/n;
%STORM-END-------------------------------------------------------------------------------------------------------------------------------


figure;
subplot(1, 2, (di-1)*2+1);
semilogy(grad_svrg, smooth(svrg-minval, 10), '-o', 'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:length(grad_svrg));
hold on;
semilogy(grad_spider, smooth(spider-minval, 10),'-*', 'Color',[0.9290 0.6940 0.1250], 'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:length(grad_spider));
semilogy(grad_scgd, smooth(scgd-minval, 10), '--','Color', [0.6350 0.0780 0.1840],  'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:length(grad_scgd));
semilogy(grad_ascpg, smooth(ascpg-minval, 10), ':', 'Color', [0.3010 0.7450 0.9330], 'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:length(grad_ascpg));
semilogy(grad_storm, smooth(storm-minval, 10), '-X', 'Color', [1 0 0], 'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:length(grad_storm));
legend('VRSC-PG','SARAH-C', 'SCGD', 'ASC-PG', 'STORM-C');
%xlim([0, 50]);
xlabel('Grads Calculation/n');
ylabel('Objective Value Gap');
title('t-SNE on MNist Dataset')
hold off;

subplot(1, 2, (di-1)*2+2);
semilogy(grad_svrg, smooth(norm_svrg, 10), '-Vb', grad_spider, smooth(norm_spider, 10), '-or', grad_scgd, smooth(norm_scgd, 10), '-V', grad_ascpg, smooth(norm_ascpg, 10), '-o', grad_storm, smooth(norm_storm, 10), '-X');
legend('VRSC-PG','SARAH-C', 'SCGD', 'ASC-PG', 'STORM-C');
xlabel('Grads Calculation/n');
ylabel('Gradient Norm');
%xlim([0, 50]);
title('t-SNE on MNist Dataset');




%plot(grad_svrg, svrg, 'b', grad_spider,spider, 'r', grad_scgd, scgd, 'g');
%legend('Svrg', 'Spider-A', 'SCGD');
%xlabel('Num Iter');
%ylabel('Objective Value');
%title('Best tuned lrscgd=1e-3, lrsvrg=5e-3, lrspider=1e-2, iters=50, A=10, B=10, eps=1e-2, *10*0.8^{epoch}')


function [dist_matrix] = Dist(data, sig)
    if sig == 0
        dist_matrix = exp(-squareform(pdist(data, 'euclidean')).^2);        
    else
        dist_matrix = exp(-squareform(pdist(data, 'euclidean')).^2./(2 * sig^2));
    end
end
