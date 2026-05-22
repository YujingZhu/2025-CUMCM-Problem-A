%% 问题2：单无人机单烟幕优化（FY1 对 M1，严格口径·仅采顶/底圆周）
% 2025年高教社杯全国大学生数学建模竞赛A题
% 目标：优化 FY1 的航向角、速度、投放点与起爆点，使对 M1 的“完全遮蔽”时间最大
% 口径：仅采“圆柱的顶/底圆的圆周边缘点”，逐点做线段-球相交；所有圆周点都被挡才算遮蔽
% 物理：FY1 等高匀速直线；弹体脱离后水平同 UAV、竖直自由落体；起爆后云团水平不动，竖直以 3 m/s 下沉，20 s 有效

clear; clc; close all;

%% 参数定义（题面）
g    = 9.81;               % 重力加速度 (m/s^2)
v_m  = 300;               % 导弹速度 (m/s)
v_s  = 3;                 % 烟幕下沉速度 (m/s)
R_s  = 10;                % 烟幕有效半径 (m)
T_eff= 20;                % 起爆后有效时间 (s)
R_t  = 7;                 % 真目标底圆半径 (m)
H_t  = 10;                % 真目标高度 (m)
dt   = 0.01;              % 时间步长 (s) —— 越小越准，稍慢

% 位置定义（题面坐标系）
O   = [0, 0, 0];          % 假目标原点
C_t = [0, 200, 5];        % 真目标几何中心（圆柱中心，高度 H_t/2 = 5 m）
M0  = [20000, 0, 2000];   % M1 初始位置
F0  = [12000, 1400, 1400];   % FY1 初始位置（等高飞）

% 导弹运动：M1 沿直线指向原点
u_m   = -M0 / norm(M0);   % 指向原点的单位向量
T_max = norm(M0) / v_m;   % 理论最大飞行时间 ≈ 66.67 s

%% PSO 优化参数（可按耗时/精度调）
N_particles = 100;        % 粒子数
MaxIter     = 300;        % 最大迭代
w  = 0.7;                 % 惯性权重
c1 = 1.5;                 % 个体学习因子
c2 = 1.5;                 % 社会学习因子

% 决策变量：x = [theta1, v_u, t_rel, t_fuse]
% theta1: 航向角(rad), v_u: FY1 速度(m/s), t_rel: 投放延迟(s), t_fuse: 引信时间(s)
lb = [0,     70,   0,   0.1];
ub = [2*pi, 140,  30,   8.0];

fprintf('开始 PSO 优化...\n');
fprintf('粒子数: %d, 迭代: %d\n', N_particles, MaxIter);

%% 初始化粒子群
rng('default');
X = lb + (ub - lb) .* rand(N_particles, 4);  % 初始位置
V = 0.1 * (ub - lb) .* rand(N_particles, 4); % 初始速度
pBest    = X;                                 % 个体最优位置
pBestVal = inf(N_particles, 1);               % 个体最优值（适应度：负遮蔽时间）
gBest    = X(1,:);                            % 全局最优位置
gBestVal = inf;                               % 全局最优值

%% PSO 主循环
for iter = 1:MaxIter
    for p = 1:N_particles
        % 适应度：负的完全遮蔽时长（越小越好）
        fitness = objectiveFunction(X(p,:), M0, F0, C_t, u_m, g, v_m, v_s, R_s, R_t, H_t, T_eff, dt);

        if fitness < pBestVal(p)
            pBest(p,:)  = X(p,:);
            pBestVal(p) = fitness;
        end
        if fitness < gBestVal
            gBest    = X(p,:);
            gBestVal = fitness;
        end
    end

    % 速度与位置更新
    for p = 1:N_particles
        r1 = rand(1,4); r2 = rand(1,4);
        V(p,:) = w*V(p,:) + c1*r1.*(pBest(p,:) - X(p,:)) + c2*r2.*(gBest - X(p,:));
        % 速度夹取
        Vmax = 0.2 * (ub - lb);
        V(p,:) = max(min(V(p,:), Vmax), -Vmax);
        % 位置更新与边界处理
        X(p,:) = X(p,:) + V(p,:);
        X(p,:) = max(min(X(p,:), ub), lb);
    end

    if mod(iter, 50)==0 || iter==1
        fprintf('迭代 %d: 最佳“完全遮蔽”时长 = %.3f s\n', iter, -gBestVal);
    end
end

%% 输出最优结果
fprintf('\n=== 优化完成（严格口径·圆周采样）===\n');
fprintf('最优参数：\n');
fprintf('  航向角 theta1 = %.4f rad (%.2f°)\n', gBest(1), gBest(1)*180/pi);
fprintf('  FY1 速度 v_u   = %.2f m/s\n', gBest(2));
fprintf('  投放延迟 t_rel  = %.3f s\n', gBest(3));
fprintf('  引信时间 t_fuse = %.3f s\n', gBest(4));

[fitness_star, P_drop, C_e, t_e] = objectiveFunction(gBest, M0, F0, C_t, u_m, g, v_m, v_s, R_s, R_t, H_t, T_eff, dt);
fprintf('\n关键位置：\n');
fprintf('  投放点 P_drop = (%.2f, %.2f, %.2f) m\n', P_drop);
fprintf('  起爆点 C_e    = (%.2f, %.2f, %.2f) m\n', C_e);
fprintf('  起爆时间 t_e  = %.3f s\n', t_e);
fprintf('\n最大“完全遮蔽”时长：%.3f s\n', -fitness_star);

% 可视化（可选）
visualizeResult(gBest, M0, F0, C_t, u_m, g, v_m, v_s, R_s, T_eff, dt);

%% =================== 目标函数 ===================
function [fitness, P_drop, C_e, t_e] = objectiveFunction(x, M0, F0, C_t, u_m, g, v_m, v_s, R_s, R_t, H_t, T_eff, dt)
    theta1 = x(1);      % 航向角
    v_u    = x(2);      % FY1 速
    t_rel  = x(3);      % 投放延迟
    t_fuse = x(4);      % 起爆延时（引信）

    % FY1 等高直线：方向在 xy 平面
    dir = [cos(theta1), sin(theta1), 0];

    % 投放点 & 投放瞬时速度
    P_drop = F0 + v_u * t_rel * dir;
    v_drop = v_u * dir;

    % 起爆点（平抛）：水平携带 v_drop，竖直自由落体
    C_e = P_drop + v_drop * t_fuse;
    C_e(3) = C_e(3) - 0.5 * g * t_fuse^2;

    % 起爆时刻
    t_e = t_rel + t_fuse;

    % 物理/可行性约束（简单惩罚）
    if C_e(3) <= 0 || t_e < 0
        fitness = 1e3;  % 大惩罚
        return;
    end

    % 遮蔽统计：严格口径（仅圆周点）
    T_cover = 0;
    t_vec = t_e:dt:(t_e + T_eff);
    for t = t_vec
        % 导弹位置（沿 u_m 匀速 v_m）
        M_pos = M0 + v_m * t * u_m;
        % 起爆后云团中心（水平不动，竖直以 v_s 下沉）
        C_cloud = C_e - [0, 0, v_s * (t - t_e)];

        if isOccluded_rims_strict(M_pos, C_t, C_cloud, R_t, H_t, R_s)
            T_cover = T_cover + dt;
        end
    end

    fitness = -T_cover;  % 取负以实现“最大化遮蔽时长”
end

%% ============== 遮蔽判定（严格口径·仅顶/底圆周） ==============
function occluded = isOccluded_rims_strict(M_pos, C_t_center, C_cloud, R_t, H_t, R_s)
    % 把“几何中心”换成“底圆心”
    C_base = [C_t_center(1), C_t_center(2), C_t_center(3) - H_t/2];

    % 仅采顶/底圆周点（不采侧壁/圆盘内部）
    nTheta = 360;  % 每个圆周按 1° 采样；可调：180/240/720 ...
    rimPts = sampleCylinderRims(C_base, R_t, H_t, nTheta);  % 2*nTheta 点

    % 逐点检查：线段 M->P 与球 (C_cloud, R_s) 是否在 [0,1] 相交
    occluded = true;
    for k = 1:size(rimPts,1)
        P = rimPts(k,:);
        if ~segmentSphereIntersect(M_pos, P, C_cloud, R_s)
            occluded = false;
            return;
        end
    end
end

%% ============== 圆柱顶/底圆周采样（仅边缘） ==============
function rimPts = sampleCylinderRims(C_base, R, H, nTheta)
% 仅采圆柱“顶/底圆”的圆周边缘点（不采侧壁、不采圆盘内部）
% C_base：底圆心 [cx,cy,cz]；顶圆心 z = cz + H
    cx = C_base(1); cy = C_base(2); cz = C_base(3);
    z_bot = cz;
    z_top = cz + H;

    thetas = linspace(0, 2*pi, nTheta+1);
    thetas(end) = [];  % 去重

    rimPts = zeros(2*nTheta, 3);
    for i = 1:nTheta
        cth = cos(thetas(i)); sth = sin(thetas(i));
        % 底圆边缘
        rimPts(2*i-1, :) = [cx + R*cth, cy + R*sth, z_bot];
        % 顶圆边缘
        rimPts(2*i,   :) = [cx + R*cth, cy + R*sth, z_top];
    end
end

%% ============== 线段-球相交（充要判定，二次方程） ==============
function hit = segmentSphereIntersect(M, P, C, r)
% 线段 M->P 与球 (C,r) 是否在 [0,1] 内有交点
% X(t) = M + t*(P-M),  t ∈ [0,1]
% |X(t)-C|^2 = r^2  =>  A t^2 + B t + Q = 0
    d = P - M;              % 方向
    f = M - C;              % 相对球心向量
    A = dot(d, d);
    if A <= 1e-16
        % 退化：M ≈ P，直接判是否在球内
        hit = (dot(f,f) <= r*r + 1e-12);
        return;
    end
    B = 2.0 * dot(f, d);
    Q = dot(f, f) - r*r;
    D = B*B - 4*A*Q;
    if D <= 0
        hit = false;
        return;
    end
    sqrtD = sqrt(D);
    t1 = (-B - sqrtD) / (2*A);
    t2 = (-B + sqrtD) / (2*A);
    t_enter = min(t1, t2);
    t_exit  = max(t1, t2);
    % 与线段相交：存在 t ∈ [0,1]，且 t_enter ≤ 1, t_exit ≥ 0（区间有重叠）
    hit = (t_exit >= 0.0) && (t_enter <= 1.0);
end

%% ============== 可视化（可选） ==============
function visualizeResult(gBest, M0, F0, C_t, u_m, g, v_m, v_s, R_s, T_eff, dt)
    figure('Position', [100, 100, 1200, 800]);

    % 取最优参数并复算关键点
    [~, P_drop, C_e, t_e] = objectiveFunction(gBest, M0, F0, C_t, u_m, g, v_m, v_s, R_s, 7, 10, T_eff, dt);

    % 轨迹
    t_missile = 0:0.1:70;
    M_traj = M0 + (v_m .* (t_missile')) .* u_m;

    theta1 = gBest(1); v_u = gBest(2);
    dir = [cos(theta1), sin(theta1), 0];
    t_drone = 0:0.1:gBest(3);
    F_traj = F0 + (v_u .* (t_drone')) .* dir;

    % 绘图
    plot3(M_traj(:,1), M_traj(:,2), M_traj(:,3), 'r-', 'LineWidth', 2); hold on;
    plot3(F_traj(:,1), F_traj(:,2), F_traj(:,3), 'b-', 'LineWidth', 2);

    plot3(M0(1), M0(2), M0(3), 'ro', 'MarkerFaceColor','r', 'DisplayName','M1起点');
    plot3(F0(1), F0(2), F0(3), 'bo', 'MarkerFaceColor','b', 'DisplayName','FY1起点');
    plot3(C_t(1), C_t(2), C_t(3), 'go', 'MarkerFaceColor','g', 'DisplayName','真目标中心');
    plot3(P_drop(1), P_drop(2), P_drop(3), 'mo', 'MarkerFaceColor','m', 'DisplayName','投放点');
    plot3(C_e(1), C_e(2), C_e(3), 'co', 'MarkerFaceColor','c', 'DisplayName','起爆点');

    % 起爆烟幕球
    [xs, ys, zs] = sphere(40);
    surf(C_e(1)+R_s*xs, C_e(2)+R_s*ys, C_e(3)+R_s*zs, ...
         'FaceAlpha',0.25,'EdgeColor','none','DisplayName','烟幕球（起爆）');

    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title('FY1 对 M1 · 烟幕干扰最优策略（严格口径·仅顶/底圆周）');
    legend({'导弹轨迹','无人机轨迹','M1起点','FY1起点','真目标中心','投放点','起爆点','烟幕球'}, ...
           'Location','northeastoutside');
    grid on; axis equal; view(45,30);
end

