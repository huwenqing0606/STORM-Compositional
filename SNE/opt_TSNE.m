%%Optimization For STORM-Compositional over stochastic neighbor embedding

%Due to complicacies in the calculations of SNE problem (involving 3rd-order tensors), 
%we have redeveloped the whole optimization part including all necessary derivative calculations. 
%Our derivative calculations are incapsulated into several functions at the end of this piece of code.
%We feel there are heavy inconsistencies in the derivative calculations for the original SARAH-C code. 
%However, for the code posted at http://github.com/angeoz/SCGD, we still try to keep intact as much as we can.

%author: Wenqing Hu (Missouri S&T)

function [resu_obj, resu_cal, resu_norm] = opt_TSNE(data, config)

w = zeros(size(data.P, 1), config.m) + [1:size(data.P,1)]'/size(data.P,1);
        %w=(w_1//...//w_n) each w_i is d-dimensional row vector
w_t = w;
w_fix = w;
x = 0;


resu_obj = zeros(1,config.max_epochs);
grad_cal = 0;
resu_cal = zeros(1, config.max_epochs);
resu_norm = zeros(1, config.max_epochs);


%% initialize g, G, F by minibatch-sampling the initial batches
if config.opt == 4
%STORM-BEGIN-------------------------------------------------------------------------------------------------------------------------------
    %initialize STORM g, G, F
    [g, G, F] = STORM_GD(data, w, config.STORM_initial_bs, config.STORM_ifreplace);
%STORM-END-------------------------------------------------------------------------------------------------------------------------------
else
    %% initialize y
    [g, G, F] = GD(data, w, config.outer_bs);
    y = g;
end

count = 0;


if config.opt == 4
%STORM-BEGIN-------------------------------------------------------------------------------------------------------------------------------
    %opt=4 indicates the STORM algorithm, number of iteration=max_epoch*max_iters
    for epoch = 1:config.max_epochs
        tic;
        for iter = 1:config.STORM_max_inner_iters
            [g, G, F] = STORM(data, w, w_t, g, G, F, config.STORM_loop_bs_g, config.STORM_loop_bs_G, config.STORM_loop_bs_F, config.STORM_a_g, config.STORM_a_G, config.STORM_a_F, config.STORM_ifreplace);
            w_t = w;
            w_tilde = w - config.STORM_lr * F;
            if config.STORM_ifnormalization == 1
                gamma = min(1/2, config.STORM_eps/norm(F)); %normalization step in STORM-Compositional Algorithm
                w = w + gamma * (w_tilde-w);
            else
                w = w_tilde;
            end
            x = x + w;
            count = count + 1; 
            grad_cal = grad_cal + config.STORM_loop_bs_g + config.STORM_loop_bs_G + config.STORM_loop_bs_F; %store how many gradient queries are taken
            if config.l1 ~= 0
                w = sign(w).* max(0, abs(w)-config.l1);
            end
        end
        norm_F = norm(F);
        xresu = x/count;
        if config.m == 1
            [obj, l2] = compute_tsne(data, xresu, config);
        else
            [obj, l2] = compute_tsne(data, w, config);
        end
        %print the IFO, gradient norms and objective value
        fprintf('STORM-C: epoch %d, IFO = %d, grad norm = %.4f, objective value = %.4f \n', epoch, grad_cal, norm_F, obj); 
        resu_obj(epoch) = obj;
        resu_norm(epoch) = norm_F;
        resu_cal(epoch) = grad_cal; 
    end
%STORM-END-------------------------------------------------------------------------------------------------------------------------------
else
for epoch = 1:config.max_epochs
    tic;
    %Outer loop update w;
    if (config.opt == 1)||(config.opt == 2)
        [g, G, F] = GD(data, w, config.outer_bs);
    elseif (config.opt == 0)
        [g, G, F, y] = SCGD(data, w, y, config.outer_bs, config.beta);
    elseif (config.opt == 3)
        [g, G, F, y] = ASCPG(data, w, w_fix, y, config.outer_bs, config.beta);
    end
    w_fix = w;
    w = w - config.lr * F;
    x = x + w;
    count = count + 1;
    if config.l1 ~= 0
		w = sign(w).* max(0, abs(w)-config.l1);
    end
    g_fix = g; G_fix = G; F_fix = F;
    norm_F = norm(F);
    grad_cal = grad_cal + config.outer_bs * 2 + 1;
    xresu = x/count;
    if config.mm == 1
        [obj, l2] = compute_tsne(data, xresu, config);
    else
        [obj, l2] = compute_tsne(data, w, config);
    end
        
    resu_obj(epoch) = obj;
    resu_norm(epoch) = norm_F;
    resu_cal(epoch) = grad_cal; 
    %print the IFO, gradient norms and objective value
    if config.opt == 2
        fprintf('SARAH-C: epoch %d, IFO = %d, grad norm = %.4f, objective value = %.4f \n', epoch, grad_cal, norm_F, obj);
    elseif config.opt == 1
        fprintf('VRSC-PG: epoch %d, IFO = %d, grad norm = %.4f, objective value = %.4f \n', epoch, grad_cal, norm_F, obj);
    elseif config.opt == 0
        fprintf('SCGD: epoch %d, IFO = %d, grad norm = %.4f, objective value = %.4f \n', epoch, grad_cal, norm_F, obj);
    else  %config.opt == 3
        fprintf('ASC-PG: epoch %d, IFO = %d, grad norm = %.4f, objective value = %.4f \n', epoch, grad_cal, norm_F, obj);
    end
    for iter = 1:config.max_iters
        if config.opt == 2
            %opt == 2 indicates SARAH_C algorithm
            %Inner loop update w;
            [g, G, F] = SARAH(data, w, w_t, g, G, F, config.inner_bs);
            grad_cal = grad_cal + config.inner_bs * 2 + 1; 
            w_t = w;
            w = w - config.lr * F;
            x = x + w;
            count = count + 1;
        elseif config.opt == 1
            [g, G, F] = SARAH(data, w, w_fix, g_fix, G_fix, F_fix, config.inner_bs);
            grad_cal = grad_cal + config.inner_bs * 2 + 1;
            w = w - config.lr * F;
            x = x + w;
            count = count + 1;
        else
            continue;
        end
        if config.l1 ~= 0
            w = sign(w).* max(0, abs(w)-config.l1);
        end
    end
end
end
end
    
function [g, G, F] = GD(data, w, batch_size)
    n = size(data.P, 1);
    d = size(w, 2);
    dataP = data.P;
    indexes = randperm(n);
    indexes = indexes(1:batch_size);
%% compute g
    g = compute_g(w, indexes);
%% compute G
    G = compute_G(w, indexes);
%% compute F
    [F_dev_, F_dev] = compute_F(w, dataP, g, indexes);
%% compute gradient
    gradphi_2 = compute_gradphi_2(G, F_dev);
    F = gradphi_2 + F_dev_;
end

function [g, G, F, y] = SCGD(data, w, y, batch_size, beta)
    n = size(data.P, 1);
    d = size(w, 2);
    dataP = data.P;
    indexes = randperm(n);
    indexes = indexes(1:batch_size);
%% compute g
    g = compute_g(w, indexes);
%% compute auxillary
    y = (1-beta) * y + beta * g;
%% compute G
    G = compute_G(w, indexes);
%% compute F
    [F_dev_, F_dev] = compute_F(w, dataP, y, indexes(1));
%% compute gradient
    gradphi_2 = compute_gradphi_2(G, F_dev);
    F = gradphi_2 + F_dev_;
end

function [g, G, F, y] = ASCPG(data, w, w_t, y, batch_size, beta)
    n = size(data.P, 1);
    d = size(w, 2);
    dataP = data.P;
    indexes = randperm(n);
    indexes = indexes(1:batch_size);
%% update auxillary    
    z = (1-1/beta) * w_t + (1/beta) * w;
%% compute g
    g = compute_g(z, indexes);  
        %this should be the difference between ASCPG and SCGD. 
        %It is not z but w in the original code posted at http://github.com/angeoz/SCGD
%% update auxillary
    y = (1-beta) * y + beta * g;
%% compute G
    G = compute_G(w, indexes);
%% compute F
    [F_dev_, F_dev] = compute_F(w, dataP, y, indexes(1));
%% compute gradient
    gradphi_2 = compute_gradphi_2(G, F_dev);
    F = gradphi_2 + F_dev_;
end

function [g, G, F] = SARAH(data, w, w_t, g, G, F, batch_size)
    n = size(data.P, 1);
    d = size(w, 2);
    dataP = data.P;
    indexes = randperm(n);
    indexes = indexes(1:batch_size);
%% compute g
    g_mat = compute_g(w, indexes);
    g_mat_t = compute_g(w_t, indexes);
    g_t = g;
    g = g_mat - g_mat_t + g;
%% compute G
    G_t = G;
    G_ = compute_G(w, indexes);
    G_i_t = compute_G(w_t, indexes);
    G = G_ - G_i_t + G;
%% compute F
    indexes = randperm(n);
    indexes = indexes(1);
    [F_dev_, F_dev] = compute_F(w, dataP, g, indexes);
    [F_dev_t, F_devt] = compute_F(w_t, dataP, g_t, indexes);
%% compute gradient
    grad = compute_gradphi_2(G, F_dev);
    grad_t = compute_gradphi_2(G_t, F_devt);
    F = F + grad - grad_t + F_dev_ - F_dev_t;
end

%%% The only difference between SARAH and SVRG is that in SVRG, w_t, g_t,
%%% G_t, F_t do not update in each iteration. So we don't have to write
%%% another SVRG function



%calculate the affinity matrix of input data points in a high-dimensional space
function [dist_matrix] = Dist(data, sig)
    if sig == 0
        dist_matrix = exp(-squareform(pdist(data, 'euclidean')).^2);
    else
        dist_matrix = exp(-squareform(pdist(data, 'euclidean')).^2/(2 * sig^2));
    end
end



%STORM-BEGIN-------------------------------------------------------------------------------------------------------------------------------

%The STORM gradient initialization
function [g, G, F] = STORM_GD(data, w, batch_size, ifreplace)
    n = size(data.P, 1);    
    d = size(w, 2);
    dataP = data.P;
    %Choose batch of batch size batch_size
    if ifreplace == 1
        indexes = datasample([1:n], batch_size); %sample with replacement
    else
        indexes = randperm(n);
        indexes = indexes(1:batch_size);        %sample without replacement    
    end
%% compute g
    g = compute_g(w, indexes);
%% compute G
    G = compute_G(w, indexes);
%% compute F
    [F_dev_, F_dev] = compute_F(w, dataP, g, indexes);
%% compute gradient
    gradphi_2 = compute_gradphi_2(G, F_dev);
    F = gradphi_2 + F_dev_;
end


%The STORM estimator for SNE problem
function [g, G, F] = STORM(data, w, w_t, g, G, F, batch_size_g, batch_size_G, batch_size_F, a_g, a_G, a_F, ifreplace)
    n = size(data.P, 1);
    d = size(w, 2);
    dataP = data.P;

%% compute g
    %Choose minibatch B_{t+1}^g
    if ifreplace == 1
        indexes = datasample([1:n], batch_size_g); %sample with replacement
    else
        indexes = randperm(n);
        indexes = indexes(1:batch_size_g);        %sample without replacement    
    end
    %g(x_{t+1}, B_{t+1}^g) 
    g_mat = compute_g(w, indexes);
    %g(x_t, B_{t+1}^g) 
    g_mat_t = compute_g(w_t, indexes);
    %g_t
    g_t = g;
    %%calculate g_{t+1} = (1-a_g)g_t + a_g g(x_{t+1}, B_{t+1}^g) + (1-a_g)[g(x_{t+1}, B_{t+1}^g)-g(x_t, B_{t+1}^g)]
    %                   = (1-a_g)g_t + g(x_{t+1}, B_{t+1}^g) - (1-a_g)g(x_t, B_{t+1}^g)
    g = (1-a_g)*g + g_mat - (1-a_g) * g_mat_t;

%% compute G = partial g at steps t and t+1 
    %Choose minibatch B_{t+1}^{partial g}
    if ifreplace == 1
        indexes = datasample([1:n], batch_size_G); %sample with replacement
    else
        indexes = randperm(n);
        indexes = indexes(1:batch_size_G);        %sample without replacement    
    end
    %G(x_{t+1}, B_{t+1}^{partial g})
    G_ = compute_G(w, indexes);
    %G(x_t, B_{t+1}^{partial g}
    G_t = compute_G(w_t, indexes);
    %%calculate G_{t+1} = (1-a_G)G_t + a_G G(x_{t+1}, B_{t+1}^g) + (1-a_G)[G(x_{t+1}, B_{t+1}^g)-G(x_t, B_{t+1}^g)]
    %                   = (1-a_G)G_t + G(x_{t+1}, B_{t+1}^g) - (1-a_G)G(x_t, B_{t+1}^g)
    G = (1-a_G) * G + G_ - (1-a_G) * G_t;
    
%% compute grad f at steps t and t+1
    %Choose minibatch B_{t+1}^f
    if ifreplace == 1
        indexes = datasample([1:n], batch_size_F); %sample with replacement
    else
        indexes = randperm(n);
        indexes = indexes(1:batch_size_F);        %sample without replacement    
    end
    %update grad f(x_{t+1}, B_{t+1}^f
    [F_dev_, F_dev] = compute_F(w, dataP, g, indexes);
    %calculate grad f(x_{t}, B_{t+1}^f)    
    [F_dev_t, F_devt] = compute_F(w_t, dataP, g_t, indexes);   
    
%% compute F update, gradient
    %calculate F_{t+1}=(1-a_F)F_t + a_F G^Tgrad_f(x_{t+1}, B_{t+1}^f) + (1-a_F)[G^Tgrad_f(x_{t+1}, B_{t+1}^f)- G_t^Tgrad_f(x_t, B_t)]
    %                 =(1-a_F)F_t + G^Tgrad_f(x_{t+1}, B_{t+1}^f) - (1-a_F) G_t^Tgrad_f(x_t, B_t)               
    gradphi_2 = compute_gradphi_2(G, F_dev);
    gradphi_2t = compute_gradphi_2(G_t, F_devt);

    F = (1-a_F)*F + gradphi_2 - (1-a_F)*gradphi_2t + F_dev_ - (1-a_F)*F_dev_t;
end

%calculate g_{n+j}(x, B) for j=1,2,...,n
function [g] = compute_g(data, indexes)
    n = size(data, 1);
    batch_size = length(indexes);
    DI = Dist(data, 0);
    g = sum(DI(indexes, :), 1) * n/batch_size - 1;
end

%calculate G(x,B)=1/|B| G_i(x) where G_i(x) are the matrices consisting of grad g with respect to x_1,...,x_n
function [G] = compute_G(data, indexes)
    n = size(data, 1);
    d = size(data, 2);
    DI = Dist(data, 0);
    batch_size = length(indexes);
    G = zeros(d, n, n);
    for l =1:n
        mat = zeros(d, n);
        for batchindex = 1:batch_size
            mat_batchindex = zeros(d, n);
            mat_batchindex(:, indexes(batchindex)) = 2*n*(data(l, :) - data(indexes(batchindex), :))'*DI(l, indexes(batchindex));
            mat_batchindex(:, l) = -2*n*(data(l, :) - data(indexes(batchindex), :))'*DI(l, indexes(batchindex));
            mat = mat + mat_batchindex;
        end
        G(:, :, l) = mat / batch_size;
    end
end

%calculate grad f^{(1)}(y, B) and grad f^{(2)}(y, B)
function [F_1, F_2] = compute_F(data, dataP, g, indexes);
    n = size(data, 1);
    d = size(data, 2);
    batch_size = length(indexes);
    F_1 = zeros(n, d);
    for j=1:batch_size
       sample_F = indexes(j);
       F_1_j = zeros(n, d);
       for l=1:n
           if l~=sample_F
               F_1_j(l, :) = 2 * n * (data(l, :) - data(sample_F, :))* dataP(sample_F, l);
           else
               F_1_j(l, :) = -2 * n * (data - data(sample_F, :))' * dataP(:, sample_F);
           end
       end
       F_1 = F_1 + F_1_j;
    end    
   
    F_1  = F_1  / batch_size;
    
    F_2 = (sum(dataP(indexes, :), 1) * n .* (1./g))'/batch_size;

end

%calculate the second part of grad f
function [gradphi_2] = compute_gradphi_2(G, F_2)
    n = size(G, 2);
    d = size(G, 1);
    gradphi_2 = zeros(n, d);
    for i=1:n
        gradphi_2(i, :) = (G(:, :, i) * F_2)';
    end
end    

%STORM-END-------------------------------------------------------------------------------------------------------------------------------
