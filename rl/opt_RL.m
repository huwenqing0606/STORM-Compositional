function [resu_obj, resu_cal, resu_norm] = opt_RL(data, config)


w = zeros(1, size(data.F,2));
w_t = w;
x = 0;
w_fix = w;


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
            w_tilde = w - config.STORM_lr * F';
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
            [obj, l2] = compute_obj(data, xresu, config);
        else
            [obj, l2] = compute_obj(data, w, config);
        end
        fprintf('STORM-C: epoch %d, grad norm = %.4f, objective value = %.4f \n', epoch, norm_F, obj); 
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
    w = w - config.lr * F';
    x = x + w;
    count = count + 1;
    if config.l1 ~= 0
		w = sign(w).* max(0, abs(w)-config.l1);
    end
    g_fix = g; G_fix = G; F_fix = F;
    norm_F = norm(F);
    grad_cal = grad_cal + config.outer_bs * 3;
    xresu = x/count;
    if config.m == 1
        [obj, l2] = compute_obj(data, xresu, config);
    else
        [obj, l2] = compute_obj(data, w, config);
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
            w = w - config.lr * F';
            x = x + w;
            count = count + 1;
        elseif config.opt == 1
            [g, G, F] = SARAH(data, w, w_fix, g_fix, G_fix, F_fix, config.inner_bs);
            grad_cal = grad_cal + config.inner_bs * 2 + 1;
            w = w - config.lr * F';
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
    dataF = data.F;
    dataP = data.P;
    dataR = data.R;
    n = size(dataF, 1);
    d = size(dataF, 2);
    indexes = randperm(n);
    indexes = indexes(1:batch_size);
%% compute g
    g_1 = dataF(:,:) * w';
    g_2 = sum(dataP(:, indexes).* (dataR(:, indexes) + 0.95 * (dataF(indexes, :) * w')'), 2)./sum(dataP(:, indexes), 2);
    g = zeros(1, 2*n);
    g(1:2:end) = g_1;
    g(2:2:end) = g_2;
%% compute G
    G_1 = dataF';
    G_2 = (dataP(:, indexes) * 0.95 * dataF(indexes, :)./sum(dataP(:, indexes), 2))';
    G = zeros(d, 2*n);
    G(:, 1:2:end) = G_1;
    G(:, 2:2:end) = G_2;
%% compute F
    indexes = indexes(1);
    indexes1 = 2 * indexes - 1;
    indexes2 = 2* indexes;
    indexes = [indexes1, indexes2];
    
    mid = 2 * (g(1:2:end) - g(2:2:end));
    F = zeros(2*n, 1);
    F(1:2:end) = mid;
    F(2:2:end) = -mid;
%% compute gradient
    F = G(:, indexes) * F(indexes);
end

function [g, G, F, y] = SCGD(data, w, y, batch_size, beta)
    dataF = data.F;
    dataP = data.P;
    dataR = data.R;
    n = size(dataF, 1);
    d = size(dataF, 2);
    indexes = randperm(n);
    indexes = indexes(1:batch_size);
%% compute g
    g_1 = dataF(:,:) * w';
    g_2 = sum(dataP(:, indexes).* (dataR(:, indexes) + 0.95 * (dataF(indexes, :) * w')'), 2)./sum(dataP(:, indexes), 2);
    g = zeros(1, 2*n);
    g(1:2:end) = g_1;
    g(2:2:end) = g_2;
%% compute auxillary
    y = (1-beta) * y + beta * g;    G_1 = dataF';
%% compute G
    G_1 = dataF';
    G_2 = (dataP(:, indexes) * 0.95 * dataF(indexes, :)./sum(dataP(:, indexes), 2))';
    G = zeros(d, 2*n);
    G(:, 1:2:end) = G_1;
    G(:, 2:2:end) = G_2;
%% compute F
    indexes = indexes(1);
    indexes1 = 2 * indexes - 1;
    indexes2 = 2* indexes;
    indexes = [indexes1, indexes2];
    
    mid = 2 * (y(1:2:end) - y(2:2:end));
    F = zeros(2*n, 1);
    F(1:2:end) = mid;
    F(2:2:end) = -mid;
%% compute gradient
    F = G(:, indexes) * F(indexes);
end

function [g, G, F, y] = ASCPG(data, w, w_t, y, batch_size, beta)
    dataF = data.F;
    dataP = data.P;
    dataR = data.R;
    n = size(dataF, 1);
    d = size(dataF, 2);
    indexes = randperm(n);
    indexes = indexes(1:batch_size);
%% update auxillary    
    z = (1-1/beta) * w_t + (1/beta)*w;
%% compute g
    g_1 = dataF(:,:) * z';
    g_2 = sum(dataP(:, indexes).* (dataR(:, indexes) + 0.95 * (dataF(indexes, :) * z')'), 2)./sum(dataP(:, indexes), 2);
    g = zeros(1, 2*n);
    g(1:2:end) = g_1;
    g(2:2:end) = g_2;
    
%% update auxillary
    y = (1-beta) * y + beta * g;
%% compute G    
    G_1 = dataF';
    G_2 = (dataP(:, indexes) * 0.95 * dataF(indexes, :)./sum(dataP(:, indexes), 2))';
    G = zeros(d, 2*n);
    G(:, 1:2:end) = G_1;
    G(:, 2:2:end) = G_2;
%% compute F
    indexes = indexes(1);
    indexes1 = 2 * indexes - 1;
    indexes2 = 2* indexes;
    indexes = [indexes1, indexes2];
    
    mid = 2 * (y(1:2:end) - y(2:2:end));
    F = zeros(2*n, 1);
    F(1:2:end) = mid;
    F(2:2:end) = -mid;
%% compute gradient
    F = G(:, indexes) * F(indexes);

end

function [g, G, F] = SARAH(data, w, w_t, g, G, F, batch_size)
    dataF = data.F;
    dataP = data.P;
    dataR = data.R;
    n = size(dataF, 1);
    d = size(dataF, 2);
    indexes = randperm(n);
    indexes = indexes(1:batch_size);
%% compute g
    g_1 = dataF(:,:) * w';
    g_2 = sum(dataP(:, indexes).* (dataR(:, indexes) + 0.95 * (dataF(indexes, :) * w')'), 2)./sum(dataP(:, indexes), 2);
    g_mat = zeros(1, 2*n);
    g_mat(1:2:end) = g_1;
    g_mat(2:2:end) = g_2;
    
    g_1_t = dataF(:,:) * w_t';
    g_2_t = sum(dataP(:, indexes).* (dataR(:, indexes) + 0.95 * (dataF(indexes, :) * w_t')'), 2)./sum(dataP(:, indexes), 2);
    g_mat_t = zeros(1, 2*n);
    g_mat_t(1:2:end) = g_1_t;
    g_mat_t(2:2:end) = g_2_t;
    g_t = g;
    g = g_mat - g_mat_t + g;
%% compute G
    G_1 = dataF';
    G_2 = (dataP(:, indexes) * 0.95 * dataF(indexes, :)./sum(dataP(:, indexes), 2))';
    G_ = zeros(d, 2*n);
    G_(:, 1:2:end) = G_1;
    G_(:, 2:2:end) = G_2;
    
    
    G_1_t = dataF';
    G_2_t = (dataP(:, indexes) * 0.95 * dataF(indexes, :)./sum(dataP(:, indexes), 2))';
    G_t_ = zeros(d, 2*n);
    G_t_(:, 1:2:end) = G_1_t;
    G_t_(:, 2:2:end) = G_2_t;
    
    G_t = G;
    %G = G_ + G_t_ + G; %was original SARAH-C code, but should be -G_t_, an error?
    G = G_ - G_t_ + G; %should be the correct case
%% compute F
    indexes = indexes(1);
    indexes1 = 2 * indexes - 1;
    indexes2 = 2* indexes;
    indexes = [indexes1, indexes2];
    
    mid = 2 * (g(1:2:end) - g(2:2:end));
    F_ = zeros(2*n, 1);
    F_(1:2:end) = mid;
    F_(2:2:end) = -mid;
    
    mid_t = 2 * (g_t(1:2:end) - g_t(2:2:end));
    F_t_ = zeros(2*n, 1);
    F_t_(1:2:end) = mid_t;
    F_t_(2:2:end) = -mid_t;

%% compute gradient
    F = F + G(:, indexes) * F_(indexes) - G_t(:, indexes) * F_t_(indexes);
end

%%% The only difference between SARAH and SVRG is that in SVRG, w_t, g_t,
%%% G_t, F_t do not update in each iteration. So we don't have to write
%%% another SVRG function




%STORM-BEGIN-------------------------------------------------------------------------------------------------------------------------------

%The STORM gradient initialization
function [g, G, F] = STORM_GD(data, w, batch_size, ifreplace)
    dataF = data.F;
    dataP = data.P;
    dataR = data.R;
    n = size(dataF, 1);
    d = size(dataF, 2);
    %Choose batch of batch size batch_size
    if ifreplace == 1
        indexes = datasample([1:n], batch_size); %sample with replacement
    else
        indexes = randperm(n);
        indexes = indexes(1:batch_size);        %sample without replacement    
    end
%% compute g
    g_1 = dataF(:,:) * w';
    g_2 = sum(dataP(:, indexes).* (dataR(:, indexes) + 0.95 * (dataF(indexes, :) * w')'), 2)./sum(dataP(:, indexes), 2);
    g = zeros(1, 2*n);
    g(1:2:end) = g_1;
    g(2:2:end) = g_2;
%% compute G
    G_1 = dataF';
    G_2 = (dataP(:, indexes) * 0.95 * dataF(indexes, :)./sum(dataP(:, indexes), 2))';
    G = zeros(d, 2*n);
    G(:, 1:2:end) = G_1;
    G(:, 2:2:end) = G_2;
%% compute F
    indexes = indexes(1);
    indexes1 = 2 * indexes - 1;
    indexes2 = 2* indexes;
    indexes = [indexes1, indexes2];
    
    mid = 2 * (g(1:2:end) - g(2:2:end));
    F = zeros(2*n, 1);
    F(1:2:end) = mid;
    F(2:2:end) = -mid;
%% compute gradient
    F = G(:, indexes) * F(indexes);
end


%The STORM estimator for reinforcement learning problem
function [g, G, F] = STORM(data, w, w_t, g, G, F, batch_size_g, batch_size_G, batch_size_F, a_g, a_G, a_F, ifreplace)
    dataF = data.F;
    dataP = data.P;
    dataR = data.R;
    n = size(dataF, 1);
    d = size(dataF, 2);
%% compute g
    %Choose minibatch B_{t+1}^g
    if ifreplace == 1
        indexes = datasample([1:n], batch_size_g); %sample with replacement
    else
        indexes = randperm(n);
        indexes = indexes(1:batch_size_g);        %sample without replacement    
    end
    %g(x_{t+1}, B_{t+1}^g) 
    g_1 = dataF(:,:) * w';
    g_2 = sum(dataP(:, indexes).* (dataR(:, indexes) + 0.95 * (dataF(indexes, :) * w')'), 2)./sum(dataP(:, indexes), 2);
    g_mat = zeros(1, 2*n);
    g_mat(1:2:end) = g_1;
    g_mat(2:2:end) = g_2;
    %g(x_t, B_{t+1}^g) 
    g_1_t = dataF(:,:) * w_t';
    g_2_t = sum(dataP(:, indexes).* (dataR(:, indexes) + 0.95 * (dataF(indexes, :) * w_t')'), 2)./sum(dataP(:, indexes), 2);
    g_mat_t = zeros(1, 2*n);
    g_mat_t(1:2:end) = g_1_t;
    g_mat_t(2:2:end) = g_2_t;
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
    G_1 = dataF';
    G_2 = (dataP(:, indexes) * 0.95 * dataF(indexes, :)./sum(dataP(:, indexes), 2))';
    G_ = zeros(d, 2*n);
    G_(:, 1:2:end) = G_1;
    G_(:, 2:2:end) = G_2;
    %G(x_t, B_{t+1}^{partial g}
    G_1_t = dataF';
    G_2_t = (dataP(:, indexes) * 0.95 * dataF(indexes, :)./sum(dataP(:, indexes), 2))';
    G_t_ = zeros(d, 2*n);
    G_t_(:, 1:2:end) = G_1_t;
    G_t_(:, 2:2:end) = G_2_t;
    %record G_t
    G_t = G;
    %%calculate G_{t+1} = (1-a_G)G_t + a_G G(x_{t+1}, B_{t+1}^g) + (1-a_G)[G(x_{t+1}, B_{t+1}^g)-G(x_t, B_{t+1}^g)]
    %                   = (1-a_G)G_t + G(x_{t+1}, B_{t+1}^g) - (1-a_G)G(x_t, B_{t+1}^g)
    G = (1-a_G) * G + G_ - (1-a_G) * G_t_;
    
%% compute grad f at steps t and t+1
    %Choose minibatch B_{t+1}^f
    if ifreplace == 1
        indexes = datasample([1:n], batch_size_F); %sample with replacement
    else
        indexes = randperm(n);
        indexes = indexes(1:batch_size_F);        %sample without replacement    
    end
    indexes = indexes(1);
    indexes1 = 2 * indexes - 1;
    indexes2 = 2* indexes;
    indexes = [indexes1, indexes2];
    
    mid = 2 * (g(1:2:end) - g(2:2:end));
    F_ = zeros(2*n, 1);
    F_(1:2:end) = mid;
    F_(2:2:end) = -mid;
    
    mid_t = 2 * (g_t(1:2:end) - g_t(2:2:end));
    F_t_ = zeros(2*n, 1);
    F_t_(1:2:end) = mid_t;
    F_t_(2:2:end) = -mid_t;
    
%% compute F update, gradient
    %calculate F_{t+1}=(1-a_F)F_t + a_F G^Tgrad_f(x_{t+1}, B_{t+1}^f) + (1-a_F)[G^Tgrad_f(x_{t+1}, B_{t+1}^f)- G_t^Tgrad_f(x_t, B_t)]
    %                 =(1-a_F)F_t + G^Tgrad_f(x_{t+1}, B_{t+1}^f) - (1-a_F) G_t^Tgrad_f(x_t, B_t)               
    F = (1-a_F)*F + G(:, indexes) * F_(indexes) - (1-a_F)*(G_t(:, indexes) * F_t_(indexes));
end

%STORM-END-------------------------------------------------------------------------------------------------------------------------------

    