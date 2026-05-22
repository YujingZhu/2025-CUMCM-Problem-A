clear; clc;
% 有时结合退火看看两者差距？
% 固定参数 
g = 9.81; v_dandao = 300; v_xiazhen = 3; R_yanwu = 10; T_xiaoguo = 20;
R_mu = 7; H_mu = 10;
C_mu = [0,200,5]; M0 = [20000,0,2000]; F0 = [12000,1400,1400];
u_dandao = -M0 / norm(M0);

lb = [0, 70, 0, 0.1]; ub = [2*pi, 140, 30, 8.0]; % 范围

N_particles = 100; MaxIter = 300; w = 0.7; c1 = 1.5; c2 = 1.5;

%% 粒子群初始化
rng('default');
pos = lb + (ub-lb).*rand(N_particles,4);
V = 0.1*(ub-lb).*rand(N_particles,4);
pBest = pos; pBestVal = inf(N_particles,1);
gBest = pos(1,:); gBestVal = inf;

% 主要循环 
for iter = 1:MaxIter
    for p = 1:N_particles
        fitness = mu_biao(xuhao(p,:), M0, F0, C_mu, u_dandao, g, v_dandao, v_xiazhen, R_yanwu, R_mu, H_mu, T_xiaoguo, 0.01);
        if fitness < pBestVal(p)
            pBest(p,:) = pos(p,:); pBestVal(p) = fitness;
        end
        if fitness < gBestVal
            gBest = pos(p,:); gBestVal = fitness;
        end
    end

    for p = 1:N_particles
        r1 = rand(1,4); r2 = rand(1,4);
        V(p,:) = w*V(p,:) + c1*r1.*(pBest(p,:) - pos(p,:)) + c2*r2.*(gBest - pos(p,:));
        Vmax = 0.2*(ub-lb); V(p,:) = max(min(V(p,:),Vmax),-Vmax);
        pos(p,:) = max(min(pos(p,:) + V(p,:),ub),lb);
    end
end

% ===================== 核心目标函数 =====================
function [fitness, P_toufang, C_e, t_e] = mu_biao(x, M0, F0, C_mu, u_dandao, g, v_dandao, v_xiazhen, R_yanwu, R_mu, H_mu, T_xiaoguo, dt)
    theta = x(1); v_uav = x(2); t_toufang = x(3); t_yinqi = x(4);
    fangxiang = [cos(theta), sin(theta), 0];

    P_toufang = F0 + v_uav*t_toufang*fangxiang;
    v_toufang = v_uav*fangxiang;

    C_e = P_toufang + v_toufang*t_yinqi;
    C_e(3) = C_e(3) - 0.5*g*t_yinqi^2;
    t_e = t_toufang + t_yinqi;

    if C_e(3)<=0 || t_e<0, fitness=1e3; return; end

    t_vec = t_e:dt:(t_e + T_xiaoguo);

    occluded_flags = arrayfun(@(t) ...
        zhe_shi(M0 + v_dandao*t*u_dandao, C_mu, C_e - [0,0,v_xiazhen*(t-t_e)], R_mu, H_mu, R_yanwu), ...
        t_vec);

    T_cover = sum(occluded_flags) * dt;
    fitness = -T_cover;
end

% ===================== 遮蔽判定 =====================
function occluded = zhe_shi(M_pos, C_mu_center, C_cloud, R_mu, H_mu, R_yanwu)
    C_base = [C_mu_center(1), C_mu_center(2), C_mu_center(3)-H_mu/2];
    nTheta = 360;
    rimPts = yuan_zhou(C_base, R_mu, H_mu, nTheta);
    occluded = true;
    for k = 1:size(rimPts,1)
        if ~xian_qiu_jiao(M_pos, rimPts(k,:), C_cloud, R_yanwu)
            occluded = false; return;
        end
    end
end

% ===================== 圆周采样 =====================
function rimPts = yuan_zhou(C_base, R_mu, H_mu, nTheta)
    cx = C_base(1); cy = C_base(2); cz = C_base(3);
    z_bot = cz; z_top = cz + H_mu;
    thetas = linspace(0,2*pi,nTheta+1); thetas(end)=[];
    rimPts = zeros(2*nTheta,3);
    for i=1:nTheta
        c = cos(thetas(i)); s = sin(thetas(i));
        rimPts(2*i-1,:) = [cx + R_mu*c, cy + R_mu*s, z_bot];
        rimPts(2*i,:)   = [cx + R_mu*c, cy + R_mu*s, z_top];
    end
end

% ===================== 线段-球相交判定 =====================
function hit = xian_qiu_jiao(M, P, C, r)
    d = P-M; f = M-C; A = dot(d,d);
    if A <= 1e-16, hit = (dot(f,f) <= r*r); return; end
    B = 2*dot(f,d); Q = dot(f,f)-r^2; D = B*B - 4*A*Q;
    if D <= 0, hit = false; return; end
    sqrtD = sqrt(D);
    t1 = (-B - sqrtD)/(2*A); t2 = (-B + sqrtD)/(2*A);
    hit = (max(t1,t2) >= 0) && (min(t1,t2) <= 1);
end
