clear; clc;
%快，但主要用PSO吧
% 固定参数
g = 9.81; v_dandao = 300; v_yanwu = 3; R_yanwu = 10; T_eff = 20;
R_t = 7; H_t = 10;
C_t = [0, 200, 5]; M0 = [20000, 0, 2000];
F0 = [12000, 1400, 1400];
u_dandao = -M0 / norm(M0);
dt = 0.01; nTheta = 720;

% 边界
lb = [0, 70, 0, 0.1];
ub = [2*pi, 140, 30, 8.0];

% SA 参数
SA.MaxStages = 180; SA.L_per_stage = 220; SA.alpha = 0.93;
SA.reheat_every = 25; SA.reheat_mult = 1.4;
SA.target_acc_hi = 0.60; SA.target_acc_lo = 0.20;
SA.step_up = 1.2; SA.step_dn = 0.8; SA.Tmin = 1e-4;
rng(42);

rimPts = buildRimPoints(C_t, R_t, H_t, nTheta);
objFun = @(x) objectiveFunction(x, M0, F0, C_t, u_dandao, g, v_dandao, v_yanwu, ...
                                R_yanwu, R_t, H_t, T_eff, dt, rimPts);

x0 = lb + (ub - lb).*rand(1,4);
DeltaS = etd(x0, 30, lb, ub, objFun);
T0 = max(1e-6, -DeltaS / log(0.8));
sigma0 = 0.10*(ub - lb); sigma = sigma0;
sigma_min = 1e-4*(ub - lb); sigma_max = 0.25*(ub - lb);

f0 = objFun(x0); x = x0; fx = f0; best_solution = x; f_best = fx;

T = T0; no_improve_stages = 0;
for stage = 1:SA.MaxStages
    n_accept = 0;
    for l = 1:SA.L_per_stage
        x_new = neighbor_gauss_reflect(x, sigma, lb, ub);
        f_new = objFun(x_new);
        dF = f_new - fx;
        accept = (dF <= 0) || (rand < exp(-dF / T));
        if accept
            x = x_new; fx = f_new; n_accept = n_accept + 1;
            if fx < f_best
                best_solution = x; f_best = fx;
            end
        end
    end
    acc_rate = n_accept / SA.L_per_stage;
    if acc_rate > SA.target_acc_hi
        sigma = min(sigma_max, SA.step_up * sigma);
    elseif acc_rate < SA.target_acc_lo
        sigma = max(sigma_min, SA.step_dn * sigma);
    end
    T = max(SA.Tmin, SA.alpha * T);
    if n_accept == 0
        no_improve_stages = no_improve_stages + 1;
        if no_improve_stages >= SA.reheat_every
            T = min(T0, SA.reheat_mult * T); no_improve_stages = 0;
        end
    else
        no_improve_stages = 0;
    end
    if T <= SA.Tmin, break; end
end

function x_new = neighbor_gauss_reflect(x, sigma, lb, ub)
    x_new = x + sigma .* randn(size(x));
    for i = 1:numel(x_new)
        while x_new(i) < lb(i) || x_new(i) > ub(i)
            if x_new(i) < lb(i)
                x_new(i) = lb(i) + (lb(i) - x_new(i));
            end
            if x_new(i) > ub(i)
                x_new(i) = ub(i) - (x_new(i) - ub(i));
            end
        end
    end
end

function DeltaS = etd(x_ref, N, lb, ub, fhandle)
    f0 = fhandle(x_ref);
    deltas = zeros(N,1);
    for k = 1:N
        x_try = lb + (ub - lb).*rand(1, numel(x_ref));
        fk = fhandle(x_try);
        deltas(k) = abs(fk - f0);
    end
    DeltaS = median(deltas);
    if ~isfinite(DeltaS) || DeltaS<=0, DeltaS = 1; end
end

function hit = lineSphereIntersect(M, p, c, r)
    d = p - M; f = M - c; A = dot(d, d);
    if A <= 1e-16, hit = (dot(f,f) <= r*r); return; end
    B = 2*dot(f,d); Q = dot(f,f)-r^2; D = B^2 - 4*A*Q;
    if D <= 0, hit = false; return; end
    t1 = (-B - sqrt(D)) / (2*A); t2 = (-B + sqrt(D)) / (2*A);
    hit = (max(t1,t2) >= 0) && (min(t1,t2) <= 1);
end

function rimPts = buildRimPoints(C_t, R_t, H_t, nTheta)
    C_base = [C_t(1), C_t(2), C_t(3) - H_t/2];
    thetas = linspace(0, 2*pi, nTheta+1); thetas(end) = [];
    rimPts = zeros(2*nTheta, 3);
    for i = 1:nTheta
        c = cos(thetas(i)); s = sin(thetas(i));
        rimPts(2*i-1,:) = [C_base(1)+R_t*c, C_base(2)+R_t*s, C_base(3)];
        rimPts(2*i,:)   = [C_base(1)+R_t*c, C_base(2)+R_t*s, C_base(3)+H_t];
    end
end

[~, P_drop, C_e, t_e] = objectiveFunction(best_solution, M0, F0, C_t, u_dandao, g, v_dandao, v_yanwu, ...
                                          R_yanwu, R_t, H_t, T_eff, dt, rimPts);
fprintf('最优参数：theta=%.2f°, v=%.2f, t_rel=%.2f, t_fuse=%.2f\n', ...
        best_solution(1)*180/pi, best_solution(2), best_solution(3), best_solution(4));
fprintf('最大完全遮蔽时长：%.3f 秒\n', -f_best);

function [fitness, P_drop, C_e, t_e] = objectiveFunction(x, M0, F0, C_t, u_dandao, g, v_dandao, v_yanwu, ...
                                                         R_yanwu, R_t, H_t, T_eff, dt, rimPts)
    theta1 = x(1); v_u = x(2); t_rel = x(3); t_fuse = x(4);
    dir = [cos(theta1), sin(theta1), 0];
    P_drop = F0 + v_u * t_rel * dir;
    v_drop = v_u * dir;
    C_e = P_drop + v_drop * t_fuse; C_e(3) = C_e(3) - 0.5 * g * t_fuse^2;
    t_e = t_rel + t_fuse;
    if C_e(3) <= 0 || t_e < 0, fitness = 1e3; return; end
    T_cover = 0;
    timeVec = t_e:dt:t_e+T_eff;
    coveredTime = sum(arrayfun(@(t) ...
        irs(M0 + v_dandao*t*u_dandao, C_e - [0,0,v_yanwu*(t-t_e)], R_yanwu, rimPts), ...
        timeVec)) * dt;
    fitness = -T_cover;
end

function occluded = irs(M_pos, C_cloud, R_s, rimPts)
    occluded = true;
    for k = 1:size(rimPts,1)
        if ~lineSphereIntersect(M_pos, rimPts(k,:), C_cloud, R_s)
            occluded = false; return;
        end
    end
end
