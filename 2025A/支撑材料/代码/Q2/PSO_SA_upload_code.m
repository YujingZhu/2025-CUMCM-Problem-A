%适合多维，2小时起步
clear; clc;
%固定参数
g = 9.81; v_dandao = 300; v_xiazhen = 3; R_yanwu = 10; T_xiaoguo = 20;
R_mu = 7; H_mu = 10;
M0 = [20000,0,2000]; F0 = [17800,0,1800]; C_mu = [0,200,5];
u_dandao = -M0/norm(M0);

%PSO 设置 
N_particles = 300; MaxIter = 500;
lb = [0,70,0,0.10]; ub = [2*pi,140,30,8.00];
w_start = 0.9; w_end = 0.4; c1 = 1.7; c2 = 1.7;
accu.dt_main = 0.005; accu.refine = true; accu.dt_refine = 2e-4;
accu.nThetaRim = 1440; accu.tolerance = 1e-12; accu.touchit = true;

%初始化 
rng(2024); 
pos = lb + (ub-lb).*rand(N_particles,4);
V = 0.1*(ub-lb).*rand(N_particles,4);
pBest = pos; pBestVal = inf(N_particles,1); gBest = pos(1,:); gBestVal = inf;

%PSO 主循环
for iter = 1:MaxIter
    w = w_start + (w_end - w_start) * (iter-1)/(MaxIter-1);
    fitnessArray = zeros(N_particles,1);
    for p = 1:N_particles
        fitnessArray(p) = mu_biao(pos(p,:), M0, F0, C_mu, u_dandao, g, v_dandao, v_xiazhen, R_yanwu, R_mu, H_mu, T_xiaoguo, accu);
    end

    improveIdx = fitnessArray < pBestVal;
    pBest(improveIdx,:) = pos(improveIdx,:); pBestVal(improveIdx) = fitnessArray(improveIdx);
    [curBestVal, idx] = min(fitnessArray);
    if curBestVal < gBestVal, gBestVal = curBestVal; gBest = pos(idx,:); end

    for p = 1:N_particles
        r1 = rand(1,4); r2 = rand(1,4);
        V(p,:) = w*V(p,:) + c1*r1.*(pBest(p,:) - pos(p,:)) + c2*r2.*(gBest - pos(p,:));
        Vmax = 0.2*(ub-lb); V(p,:) = max(min(V(p,:),Vmax),-Vmax);
        pos(p,:) = max(min(pos(p,:) + V(p,:), ub), lb);
    end
end

%目标函数
function [fitness, P_toufang, C_e, t_e] = mu_biao(x, M0, F0, C_mu, u_dandao, g, v_dandao, v_xiazhen, R_yanwu, R_mu, H_mu, T_xiaoguo, accu)
theta1 = x(1); v_uav = x(2); t_toufang = x(3); t_yinqi = x(4); dir = [cos(theta1), sin(theta1), 0];
P_toufang = F0 + v_uav*t_toufang*dir; v_toufang = v_uav*dir;
C_e = P_toufang + v_toufang*t_yinqi; C_e(3) = C_e(3) - 0.5*g*t_yinqi^2; t_e = t_toufang + t_yinqi;
if C_e(3) <= 0 || t_e < 0, fitness = 1e3; return; end

t0 = t_e; t1 = t_e + T_xiaoguo; dt = accu.dt_main; T_cover = 0; t_prev = t0; oc_prev = occludedAt(t_prev);
while t_prev < t1-1e-12
    t_next = min(t_prev+dt, t1); oc_next = occludedAt(t_next);
    if oc_prev == oc_next
        if oc_prev, T_cover = T_cover + (t_next - t_prev); end
    else
        if accu.refine
            tL = t_prev; tR = t_next;
            while (tR - tL) > accu.dt_refine
                tm = (tL + tR)/2;
                if checkOcclusion(M0, v_dandao, u_dandao, C_mu, C_e, t_e, v_xiazhen, R_mu, H_mu, R_yanwu, accu, tm) == oc_prev
                    tL = tm;
                else
                    tR = tm;
                end
            end
            t_switch = (tL + tR)/2;
        else
            t_switch = 0.5*(t_prev+t_next);
        end
        if oc_prev, T_cover = T_cover + (t_switch - t_prev); else T_cover = T_cover + (t_next - t_switch); end
    end
    t_prev = t_next; oc_prev = oc_next;
end

fitness = -T_cover;

function oc = occludedAt(t)
    M_pos = M0 + v_dandao*t*u_dandao;
    C_cloud = C_e - [0,0,v_xiazhen*(t-t_e)];
    oc = checkCylinderOcclusion(M_pos, C_mu, C_cloud, R_mu, H_mu, R_yanwu, accu);
end
end

%遮蔽判定
function occluded = checkCylinderOcclusion(M_pos, C_mu_center, C_cloud, R_mu, H_mu, R_s, accu)
C_base = [C_mu_center(1), C_mu_center(2), C_mu_center(3)-H_mu/2];
rimPts = buildRimPoints(C_base, R_mu, H_mu, accu.nThetaRim);
occluded = true;
for k = 1:size(rimPts,1)
    if ~lineSphereIntersect(M_pos, rimPts(k,:), C_cloud, R_s, accu.tolerance, accu.touchit)
        occluded = false; return;
    end
end
end

%圆柱顶/底圆周采样 
function rimPts = buildRimPoints(C_base, R_mu, H_mu, nTheta)
th = linspace(0, 2*pi, nTheta+1); th(end) = [];
rimPts = zeros(2*nTheta,3);
for i = 1:nTheta
    c = cos(th(i)); s = sin(th(i));
    rimPts(2*i-1,:) = [C_base(1)+R_mu*c, C_base(2)+R_mu*s, C_base(3)];
    rimPts(2*i,:)   = [C_base(1)+R_mu*c, C_base(2)+R_mu*s, C_base(3)+H_mu];
end
end

%线段-球相交
function hit = lineSphereIntersect(M, P, C, r, tol, touchit)
d = P-M; f = M-C; A = dot(d,d);
if A <= 1e-16, hit = (dot(f,f) <= r*r); return; end
B = 2*dot(f,d); Q = dot(f,f)-r^2; D = B^2-4*A*Q;
if D < 0
    if touchit && abs(D) <= tol*(1+abs(B)+abs(Q)+A)
        t = -B/(2*A); hit = (t >= -1e-12) && (t <= 1+1e-12);
    else
        hit = false;
    end
    return;
end
sqrtD = sqrt(max(D,0));
t1 = (-B - sqrtD)/(2*A); t2 = (-B + sqrtD)/(2*A);
t_enter = min(t1,t2); t_exit = max(t1,t2);
hit = (t_exit >= -1e-12) && (t_enter <= 1+1e-12);
end
