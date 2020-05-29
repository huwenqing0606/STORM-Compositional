%%Optimization For STORM-Compositional over portfolio optimization problem

%code between separating dashed lines are for STORM
%code else where are adapted from SARAH-Compositional to make a comparison

%author: Wenqing Hu (Missouri S&T)

function [resu_obj, resu_cal, resu_norm] = opt_VR(data, config)

w_t = ones(1, size(data, 2)); %x_{t-1}
w = ones(1, size(data, 2)); %x_t
w_fix = w;
x = zeros(1, size(data, 2));


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
        fprintf('STORM-C: epoch %d, grad norm = %.4f \n', epoch, norm_F); 
        xresu = x/count;
        if config.m == 1
            [obj, l2] = compute_port(data, xresu, config);
        else
            [obj, l2] = compute_port(data, w, config);
        end
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
        grad_cal = grad_cal + config.outer_bs * 3; %store how many gradient queries are taken
        xresu = x/count;
        if config.m == 1
            [obj, l2] = compute_port(data, xresu, config);
        else
            [obj, l2] = compute_port(data, w, config);
        end
        resu_obj(epoch) = obj;
        resu_norm(epoch) = norm_F;
        resu_cal(epoch) = grad_cal; 
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
    d = length(w);
    n = size(data, 1);%The shape is n*d
    %Choose batch of batch size batch_size
    indexes = randperm(n);
    indexes = indexes(1:batch_size);
%% compute g
    g_mat = repmat(w, batch_size, 1);
    g_mat(:, d+1) = data(indexes, :) * w';
    g = mean(g_mat);
%% compute G
    G = diag(ones(d, 1));
    G(d+1, :) = mean(data(indexes, :));
%% compute F
    mid = 2 * mean(data(indexes, :) * g(1:d)')- g(d+1);
    r = mean(data(indexes, :));
    F_dev = mean(diag(2 * data(indexes, :) * g(1:d)') * data(indexes, :)) - (g(d + 1)+1) * r;
    F_dev(d+1) = -mid;
%% compute gradient
    F = F_dev * G;
end

function [g, G, F, y] = SCGD(data, w, y, batch_size, beta)
    d = length(w);
    n = size(data, 1);%The shape is n*d
    %Choose batch of batch size batch_size
    indexes = randperm(n);
    indexes = indexes(1:batch_size);
%% compute g
    g_mat = repmat(w, batch_size, 1);
    g_mat(:, d+1) = data(indexes, :) * w';
    g = mean(g_mat);
%% compute auxillary
    y = (1-beta) * y + beta * g;
%% compute G
    G = diag(ones(d, 1));
    G(d+1, :) = mean(data(indexes, :));
%% compute F
    mid = 2 * mean(data(indexes, :) * y(1:d)')- y(d+1);
    r = mean(data(indexes, :));
    F_dev = mean(diag(2 * data(indexes, :) * y(1:d)') * data(indexes, :)) - (y(d + 1)+1) * r;
    F_dev(d+1) = -mid;
%% compute gradient
    F = F_dev * G;
end

function [g, G, F, y] = ASCPG(data, w, w_t, y, batch_size, beta)
    d = length(w);
    n = size(data, 1);%The shape is n*d
    %Choose batch of batch size batch_size
    indexes = randperm(n);
    indexes = indexes(1:batch_size);
%% update auxillary    
    z = (1-1/beta) * w_t + (1/beta)*w;
%% compute g
    g_mat = repmat(z, batch_size, 1);
    g_mat(:, d+1) = data(indexes, :) * z';
    g = mean(g_mat);
    
%% update auxillary
    y = (1-beta) * y + beta * g;
%% compute G
    G = diag(ones(d, 1));
    G(d+1, :) = mean(data(indexes, :));
%% compute F
    mid = 2 * mean(data(indexes, :) * y(1:d)')- y(d+1);
    r = mean(data(indexes, :));
    F_dev = mean(diag(2 * data(indexes, :) * y(1:d)') * data(indexes, :)) - (y(d + 1)+1) * r;
    F_dev(d+1) = -mid;
%% compute gradient
    F = F_dev * G;
end

function [g, G, F] = SARAH(data, w, w_t, g, G, F, batch_size)
    d = length(w);
    n = size(data, 1);%The shape is n*d
    %Choose batch of batch size batch_size
    indexes = randperm(n);
    indexes = indexes(1:batch_size);
%% compute g
    g_mat = repmat(w, batch_size, 1);
    g_mat(:, d+1) = data(indexes, :) * w';
    
    g_mat_t = repmat(w_t, batch_size, 1);
    g_mat_t(:, d+1) = data(indexes, :) * w_t';
    g_t = g;
    g = mean(g_mat) - mean(g_mat_t) + g;
%% compute G
    G_t = G;
    %Stay unchanged
%% compute F
    indexes = randperm(n);
    indexes = indexes(1);
    mid = 2 * mean(data(indexes, :) * g(1:d)')- g(d+1);
    r = mean(data(indexes, :));
    F_dev = mean(diag(2 * data(indexes, :) * g(1:d)') * data(indexes, :)) - (g(d + 1)+1) * r;
    F_dev(d+1) = -mid;
    
    mid_t = 2 * mean(data(indexes, :) * g_t(1:d)')- g_t(d+1);
    r_t = mean(data(indexes, :));
    F_dev_t = mean(diag(2 * data(indexes, :) * g_t(1:d)') * data(indexes, :)) - (g_t(d + 1)+1) * r_t;
    F_dev_t(d+1) = -mid_t;
%% compute gradient
    F = F_dev * G - F_dev_t * G_t + F;
end

%%% The only difference between SARAH and SVRG is that in SVRG, w_t, g_t,
%%% G_t, F_t do not update in each iteration. So we don't have to write
%%% another SVRG function



%STORM-BEGIN-------------------------------------------------------------------------------------------------------------------------------

%The STORM gradient initialization
function [g, G, F] = STORM_GD(data, w, batch_size, ifreplace)
    d = length(w);
    n = size(data, 1);%The shape is n*d
    %Choose batch of batch size batch_size
    if ifreplace == 1
        indexes = datasample([1:n], batch_size); %sample with replacement
    else
        indexes = randperm(n);
        indexes = indexes(1:batch_size);        %sample without replacement    
    end
%% compute g
    g_mat = repmat(w, batch_size, 1);
    g_mat(:, d+1) = data(indexes, :) * w';
    g = mean(g_mat);
%% compute G
    G = diag(ones(d, 1));
    G(d+1, :) = mean(data(indexes, :));
%% compute F
    mid = 2 * mean(data(indexes, :) * g(1:d)')- 2*g(d+1);  
            %the factor 2 in front of g(d+1) was missed everywhere in SARAH-C code, error?
    r = mean(data(indexes, :));
    F_dev = mean(diag(2 * data(indexes, :) * g(1:d)') * data(indexes, :)) - (2*g(d + 1)+1) * r;
            %the factor 2 in front of g(d+1) was missed everywhere in SARAH-C code, error?
    F_dev(d+1) = -mid;
%% compute gradient
    F = F_dev * G;
end


%The STORM estimator for portfolio management problem
function [g, G, F] = STORM(data, w, w_t, g, G, F, batch_size_g, batch_size_G, batch_size_F, a_g, a_G, a_F, ifreplace)
    d = length(w);    %d is here N+1
    n = size(data, 1);%n is here . The shape is n*d
%% compute g
    %Choose batch B_{t+1}^g of batch size batch_size
    if ifreplace == 1
       indexes = datasample([1:n], batch_size_g); %sample with replacement
    else
       indexes = randperm(n);
       indexes = indexes(1:batch_size_g);        %sample without replacement    
    end
    %g(x_{t+1}, B_{t+1}^g) before taking the mean
    g_mat = repmat(w, batch_size_g, 1);
    g_mat(:, d+1) = data(indexes, :) * w';
    %g(x_t, B_{t+1}^g) before taking the mean
    g_mat_t = repmat(w_t, batch_size_g, 1);
    g_mat_t(:, d+1) = data(indexes, :) * w_t';
    %g_t
    g_t = g;
    %%calculate g_{t+1} = (1-a_g)g_t + a_g g(x_{t+1}, B_{t+1}^g) + (1-a_g)[g(x_{t+1}, B_{t+1}^g)-g(x_t, B_{t+1}^g)]
    %                   = (1-a_g)g_t + g(x_{t+1}, B_{t+1}^g) - (1-a_g)g(x_t, B_{t+1}^g)
    g = (1-a_g)*g + mean(g_mat) - (1-a_g) * mean(g_mat_t);
%% compute G = partial g at steps t and t+1 
    %due to the particular structure of the portfolio optimization problem they are both constant matrices
    %stay unchanged
    G_t = G;
%% compute grad f at steps t and t+1
    %Choose batch B_{t+1}^f of batch size batch_size
    if ifreplace == 1
      indexes = datasample([1:n], batch_size_F);      %sample with replacement
    else
      indexes = randperm(n);
      indexes = indexes(1:batch_size_F);             %sample without replaceemnt
    end
    %calculate grad f at step t+1
    mid = 2 * mean(data(indexes, :) * g(1:d)')- 2*g(d+1);
    r = mean(data(indexes, :));
    F_dev = mean(diag(2 * data(indexes, :) * g(1:d)') * data(indexes, :)) - (2*g(d + 1)+1) * r;
    F_dev(d+1) = -mid;
    %calculate grad f at step t
    mid_t = 2 * mean(data(indexes, :) * g_t(1:d)')- 2*g_t(d+1);
    r_t = mean(data(indexes, :));
    F_dev_t = mean(diag(2 * data(indexes, :) * g_t(1:d)') * data(indexes, :)) - (2*g_t(d + 1)+1) * r_t;
    F_dev_t(d+1) = -mid_t;
    
%% compute F update, gradient
    %calculate F_{t+1}=(1-a_F)F_t + a_F G^Tgrad_f(x_{t+1}, B_{t+1}^f) + (1-a_F)[G^Tgrad_f(x_{t+1}, B_{t+1}^f)- G_t^Tgrad_f(x_t, B_t)]
    %                 =(1-a_F)F_t + G^Tgrad_f(x_{t+1}, B_{t+1}^f) - (1-a_F) G_t^Tgrad_f(x_t, B_t)               
    F = (1-a_F) * F + F_dev * G - (1-a_F) * (F_dev_t * G_t);
end
%STORM-END-------------------------------------------------------------------------------------------------------------------------------
