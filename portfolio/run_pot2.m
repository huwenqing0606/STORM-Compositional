%%Runfile For STORM-Compositional over portfolio optimization problem

%code between separating dashed lines are for STORM, and are newly developed by us
%code else where are adapted from SARAH-Compositional and not changed (see http://github.com/angeoz/SCGD) to make a comparison
%we try to keep the original SARAH-Compositional code as much intact as we can

%author: Jiaojiao Yang (Anhui Normal University)

clearvars;

rng(1, 'twister');

%config.l1 = 5e-5;
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
%Probname = {'Asia_Pacific_ex_Japan_OP', 'Europe_OP', 'Global_ex_US_OP', 'Global_OP', 'Japan_OP', 'North_America_OP'};
%Titlename = {'Asia Pacific ex Japan OP', 'Europe OP', 'Global ex US OP', 'Global OP', 'Japan OP', 'North America OP'};
%Probname = {'Asia_Pacific_ex_Japan_ME', 'Europe_ME', 'Global_ex_US_ME', 'Global_ME', 'Japan_ME', 'North_America_ME'}; 
%Titlename = {'Asia Pacific ex Japan ME', 'Europe ME', 'Global ex US ME', 'Global ME', 'Japan ME', 'North America ME'};
%Probname = {'Asia_Pacific_ex_Japan_OP'};
%Titlename = {'Asia Pacific ex Japan OP'};
%Probname = {'Global_ex_US_OP', 'Japan_OP', 'Global_OP', 'Asia_Pacific_ex_Japan_OP', 'Europe_OP', 'North_America_OP'};
%Titlename = {'Global ex US OP', 'Japan OP', 'Global OP', 'Asia Pacific ex Japan OP', 'Europe OP', 'North America OP'};
Probname = {'data_cov_4', 'data_cov_20'};
Titlename = {'data cov 4', 'data cov 20'};
%Probname = {'ME'};
%Probname = {'Global_ex_US_OP'};
%Probname = {'Europe_OP'};
%Probname = {'North_America_OP'};
%lrlist = [[1e-3, 5e-3, 5e-3]; [1e-3, 2e-3, 5e-4]; [1e-3, 2e-3, 5e-4];[1e-3, 5e-3, 5e-4];[1e-4, 1e-3, 5e-4];[1e-3, 2e-3, 5e-4]];
%lrlist = [[1e-3, 5e-3, 5e-3]];%Asia_Pacific
%lrlist = [[5e-5, 2e-4, 2e-4]];
%lrlist = [[1e-4, 1e-2, 1.1e-2]];%Global_ex_US_OP
%lrlist = [[1e-4, 1e-3, 1e-3]];%Japan_OP
%lrlist = [2e-3, 1e-3, 1e-3, 2e-3, 1e-3, 1e-3];
lrlist = [2e-5, 2e-5];

nprob = length(Probname);
Problist = [1:nprob];
figure;
%config.m = 1;
config.m = 0;

for di = 1:length(Problist) 
    disp(Titlename(di));
    
    %% load data
    probID = Problist(di);
    name = Probname{probID};
    %load(strcat(Probname{Problist(di)},'.mat'));
    load(strcat('~/文档/work_STORM-SCGD/STORM-SCGD_source-code/portfolio/data/', Probname{Problist(di)},'.mat'));
    %load data_cov_2;
    [n, d] = size(data);
    config.lr = lrlist(di);

    rng(1);
    minval = compute_min_val(data, config);

    config.gamma = 0.95;
    config.max_iters = 20; 
    config.outer_bs = 2000;
    config.inner_bs = 5;
    config.beta = 0.9;
    config.opt = 1;

    
    config.max_epochs = 500; %STORM shares the same epoch number with other Compositional Optimization algorithms

%STORM-BEGIN-------------------------------------------------------------------------------------------------------------------------------
    
    %set do or not do normalization step in STORM, STORM-C does normalization, but we want to compare what happens if there is no normalization 
    config.STORM_ifnormalization = 0;
    %set with or without replacement in minibatch sampling in STORM, with replacement = 1, STORM-C uses with replacement sampling, but we want to compare
    config.STORM_ifreplace = 1;
    %set the STORM single loop batchsizes, learning rate and the a parameters
    config.STORM_eps = 0.1;
    config.STORM_max_inner_iters = 20; 
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
    
%STORM-END-------------------------------------------------------------------------------------------------------------------------------
    
    [svrg, grad_svrg, norm_svrg] = opt_VR(data, config);
    grad_svrg = grad_svrg/n;
    config.opt = 2;
    config.dec = 1;
    [spider, grad_spider, norm_spider] = opt_VR(data, config);
    grad_spider = grad_spider/n;
    config.dec = 0;
    config.opt = 0;
    [scgd, grad_scgd, norm_scgd] = opt_VR(data, config);
    grad_scgd = grad_scgd/n;
    config.opt = 3;
    [ascpg, grad_ascpg, norm_ascpg] = opt_VR(data, config);
    grad_ascpg = grad_ascpg/n;
    %config.opt = 3;
    %config.lambda = 1;
    %config.lr = 3e-5;
    %config.max_iters = 1;
    %[civr, grad_civr, norm_civr, norm_c] = opt_VRSCPG(data, config);
    %config.lr = 5e-4;
    %[spider1, grad_spider1, norm_spider1] = opt_VRSCPG(data, config);
    %subplot(2, 3, di);

 %STORM-BEGIN-------------------------------------------------------------------------------------------------------------------------------

    %the option of using STORM
    config.opt = 4;
    [storm, grad_storm, norm_storm] = opt_VR(data, config);
    grad_storm = grad_storm/n;
    
 %STORM-END-------------------------------------------------------------------------------------------------------------------------------


    %figure(di);
    %subplot(1, 2, 1);
    subplot(nprob, 2, (di-1)*2+1);
    semilogy(grad_svrg, smooth(svrg-minval, 10), '-.', 'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:length(grad_svrg));
    hold on;
    semilogy(grad_spider, smooth(spider-minval, 10),'-*', 'Color',[0.9290 0.6940 0.1250], 'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:length(grad_svrg));
    semilogy(grad_scgd, smooth(scgd-minval, 10), '--','Color', [0.6350 0.0780 0.1840],  'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:length(grad_svrg));
    semilogy(grad_ascpg, smooth(ascpg-minval, 10), ':', 'Color', [0.3010 0.7450 0.9330], 'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:length(grad_svrg));
    semilogy(grad_storm, smooth(storm-minval, 10), '-X', 'Color', [1 0 0], 'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:length(grad_svrg));
    legend('VRSC-PG','SARAH-C', 'SCGD', 'ASC-PG', 'STORM-C');
    xlabel('Grads Calculation/n');
    ylabel('Objective Value Gap');
    title(Titlename(di))
    hold off;

    %subplot(1, 2, 2);
    subplot(nprob, 2, (di-1)*2+2);
    semilogy(grad_svrg, smooth(norm_svrg, 10), '-Vb', grad_spider, smooth(norm_spider, 10), '-or', grad_scgd, smooth(norm_scgd, 10), '-V', grad_ascpg, smooth(norm_ascpg, 10), '-o', grad_storm, smooth(norm_storm, 10), '-X');
    legend('VRSC-PG','SARAH-C', 'SCGD', 'ASC-PG', 'STORM-C');
    xlabel('Grads Calculation');
    ylabel('Gradient Norm');
    title(Titlename(di));
    
    % 
    % 
    % 
    % 
    % 
    % config.lr = 2e-4;
    % config.beta = 0.9;
    % config.opt = 1;
    % [svrg, grad_svrg, norm_svrg] = opt_VR(data, config);
    % config.opt = 2;
    % config.dec = 1;
    % [spider, grad_spider, norm_spider] = opt_VR(data, config);
    % config.dec = 0;
    % config.opt = 0;
    % config.lr = 2e-4;
    % [scgd, grad_scgd, norm_scgd] = opt_VR(data, config);
    % config.opt = 3;
    % config.lr = 2e-4;
    % [ascpg, grad_ascpg, norm_ascpg] = opt_VR(data, config);
    % %config.opt = 3;
    % %config.lambda = 1;
    % %config.lr = 3e-5;
    % %config.max_iters = 1;
    % %[civr, grad_civr, norm_civr, norm_c] = opt_VRSCPG(data, config);
    % %config.lr = 5e-4;
    % %[spider1, grad_spider1, norm_spider1] = opt_VRSCPG(data, config);
    % figure;
    % subplot(1, 2, 1);
    % semilogy(grad_svrg, smooth(svrg-minval, 10), '-b', grad_spider, smooth(spider-minval, 10), grad_scgd, smooth(scgd-minval, 10), '-b', grad_ascpg, smooth(ascpg-minval, 10));
    % legend('VRSC-PG','SARAH-C', 'SCGD', 'ASC-PG');
    % xlabel('Grads Calculation');
    % ylabel('Objective Value Gap');
    % title('Objective Value Gap vs. Grads Calculation')
    % 
    % subplot(1, 2, 2);
    % semilogy(grad_svrg, smooth(norm_svrg, 10), '-Vb', grad_spider, smooth(norm_spider, 10), '-or', grad_scgd, smooth(norm_scgd, 10), '-V', grad_ascpg, smooth(norm_ascpg, 10), '-o');
    % legend('VRSC-PG','SARAH-C', 'SCGD', 'ASC-PG');
    % xlabel('Grads Calculation');
    % ylabel('Gradient Norm');
    % title('Gradient Norm vs. Grads Calculation')
    % 
    % 
    % 
    % 
    % 
    % 
    % 
    % config.lr = 3e-4;
    % config.beta = 0.9;
    % config.opt = 1;
    % [svrg, grad_svrg, norm_svrg] = opt_VR(data, config);
    % config.opt = 2;
    % config.dec = 1;
    % [spider, grad_spider, norm_spider] = opt_VR(data, config);
    % config.dec = 0;
    % config.opt = 0;
    % config.lr = 3e-4;
    % [scgd, grad_scgd, norm_scgd] = opt_VR(data, config);
    % config.opt = 3;
    % config.lr = 3e-4;
    % [ascpg, grad_ascpg, norm_ascpg] = opt_VR(data, config);
    % %config.opt = 3;
    % %config.lambda = 1;
    % %config.lr = 3e-5;
    % %config.max_iters = 1;
    % %[civr, grad_civr, norm_civr, norm_c] = opt_VRSCPG(data, config);
    % %config.lr = 5e-4;
    % %[spider1, grad_spider1, norm_spider1] = opt_VRSCPG(data, config);
    % figure;
    % subplot(1, 2, 1);
    % semilogy(grad_svrg, smooth(svrg-minval, 10), '-b', grad_spider, smooth(spider-minval, 10), grad_scgd, smooth(scgd-minval, 10), '-b', grad_ascpg, smooth(ascpg-minval, 10));
    % legend('VRSC-PG','SARAH-C', 'SCGD', 'ASC-PG');
    % xlabel('Grads Calculation');
    % ylabel('Objective Value Gap');
    % title('Objective Value Gap vs. Grads Calculation')
    % 
    % subplot(1, 2, 2);
    % semilogy(grad_svrg, smooth(norm_svrg, 10), '-Vb', grad_spider, smooth(norm_spider, 10), '-or', grad_scgd, smooth(norm_scgd, 10), '-V', grad_ascpg, smooth(norm_ascpg, 10), '-o');
    % legend('VRSC-PG','SARAH-C', 'SCGD', 'ASC-PG');
    % xlabel('Grads Calculation');
    % ylabel('Gradient Norm');
    % title('Gradient Norm vs. Grads Calculation')
    % 
    % 
    % 
    % 
    % 
    % 
    % 
    % 
    % config.lr = 5e-4;
    % config.beta = 0.9;
    % config.opt = 1;
    % [svrg, grad_svrg, norm_svrg] = opt_VR(data, config);
    % config.opt = 2;
    % config.dec = 1;
    % [spider, grad_spider, norm_spider] = opt_VR(data, config);
    % config.dec = 0;
    % config.opt = 0;
    % config.lr = 5e-4;
    % [scgd, grad_scgd, norm_scgd] = opt_VR(data, config);
    % config.opt = 3;
    % config.lr = 5e-4;
    % [ascpg, grad_ascpg, norm_ascpg] = opt_VR(data, config);
    % %config.opt = 3;
    % %config.lambda = 1;
    % %config.lr = 3e-5;
    % %config.max_iters = 1;
    % %[civr, grad_civr, norm_civr, norm_c] = opt_VRSCPG(data, config);
    % %config.lr = 5e-4;
    % %[spider1, grad_spider1, norm_spider1] = opt_VRSCPG(data, config);
    % figure;
    % subplot(1, 2, 1);
    % semilogy(grad_svrg, smooth(svrg-minval, 10), '-b', grad_spider, smooth(spider-minval, 10), grad_scgd, smooth(scgd-minval, 10), '-b', grad_ascpg, smooth(ascpg-minval, 10));
    % legend('VRSC-PG','SARAH-C', 'SCGD', 'ASC-PG');
    % xlabel('Grads Calculation');
    % ylabel('Objective Value Gap');
    % title('Objective Value Gap vs. Grads Calculation')
    % 
    % subplot(1, 2, 2);
    % semilogy(grad_svrg, smooth(norm_svrg, 10), '-Vb', grad_spider, smooth(norm_spider, 10), '-or', grad_scgd, smooth(norm_scgd, 10), '-V', grad_ascpg, smooth(norm_ascpg, 10), '-o');
    % legend('VRSC-PG','SARAH-C', 'SCGD', 'ASC-PG');
    % xlabel('Grads Calculation');
    % ylabel('Gradient Norm');
    % title('Gradient Norm vs. Grads Calculation')
    % 
    % 
    % 
    % 
    % 
    % 
    % 
    % config.lr = 8e-4;
    % config.beta = 0.9;
    % config.opt = 1;
    % [svrg, grad_svrg, norm_svrg] = opt_VR(data, config);
    % config.opt = 2;
    % config.dec = 1;
    % [spider, grad_spider, norm_spider] = opt_VR(data, config);
    % config.dec = 0;
    % config.opt = 0;
    % config.lr = 8e-4;
    % [scgd, grad_scgd, norm_scgd] = opt_VR(data, config);
    % config.opt = 3;
    % config.lr = 8e-4;
    % [ascpg, grad_ascpg, norm_ascpg] = opt_VR(data, config);
    % %config.opt = 3;
    % %config.lambda = 1;
    % %config.lr = 3e-5;
    % %config.max_iters = 1;
    % %[civr, grad_civr, norm_civr, norm_c] = opt_VRSCPG(data, config);
    % %config.lr = 5e-4;
    % %[spider1, grad_spider1, norm_spider1] = opt_VRSCPG(data, config);
    % figure;
    % subplot(1, 2, 1);
    % semilogy(grad_svrg, smooth(svrg-minval, 10), '-b', grad_spider, smooth(spider-minval, 10), grad_scgd, smooth(scgd-minval, 10), '-b', grad_ascpg, smooth(ascpg-minval, 10));
    % legend('VRSC-PG','SARAH-C', 'SCGD', 'ASC-PG');
    % xlabel('Grads Calculation');
    % ylabel('Objective Value Gap');
    % title('Objective Value Gap vs. Grads Calculation')
    % 
    % subplot(1, 2, 2);
    % semilogy(grad_svrg, smooth(norm_svrg, 10), '-Vb', grad_spider, smooth(norm_spider, 10), '-or', grad_scgd, smooth(norm_scgd, 10), '-V', grad_ascpg, smooth(norm_ascpg, 10), '-o');
    % legend('VRSC-PG','SARAH-C', 'SCGD', 'ASC-PG');
    % xlabel('Grads Calculation');
    % ylabel('Gradient Norm');
    % title('Gradient Norm vs. Grads Calculation')
    % %plot(grad_svrg, svrg, 'b', grad_spider,spider, 'r', grad_scgd, scgd, 'g');
    % %legend('Svrg', 'Spider-A', 'SCGD');
    % %xlabel('Num Iter');
    % %ylabel('Objective Value');
    % %title('Best tuned lrscgd=1e-3, lrsvrg=5e-3, lrspider=1e-2, iters=50, A=10, B=10, eps=1e-2, *10*0.8^{epoch}')
    
end