%% 改良模拟退火（SA）用于：问题2 · 单无人机单烟幕（严格口径·仅采顶/底圆周）
% 目标：最大化“完全遮蔽”时长（所有顶/底圆周边缘点都被烟幕挡住的累计时长）
% 决策变量：x = [theta1, v_u, t_rel, t_fuse]
% 重要说明：
%   1) 本文件是完整可运行脚本。局部函数写在文末；用“匿名函数句柄”捕获常量，避免作用域错误。
%   2) F0 请根据你使用的无人机编号检查：
%        - FY1（题面）：F0 = [17800, 0, 1800];
%        - FY2（你原脚本）：F0 = [12000, 1400, 1400];
%   3) 精度-效率开关：dt（时间步）与 nTheta（圆周采样数）。越大/越密越慢但更准。

clear; clc; close all;

%% =============== 题面与物理参数 =================
g    = 9.81;               % 重力加速度 (m/s^2)
v_m  = 300;                % 导弹速度 (m/s)
v_s  = 3;                  % 烟幕下沉速度 (m/s)
R_s  = 10;                 % 烟幕有效半径 (m)
T_eff= 20;                 % 起爆后有效时间 (s)
R_t  = 7;                  % 真目标底圆半径 (m)
H_t  = 10;                 % 真目标高度 (m)

% 位置（题面坐标系）
C_t = [0, 200, 5];         % 真目标几何中心（H_t/2 = 5）
M0  = [20000, 0, 2000];    % M1 初始位置
% === 按需选择你的无人机初始坐标 ===
% F0 = [17800, 0, 1800];   % FY1（题面）
F0  = [12000, 1400, 1400]; % FY2（你原脚本）——若解 FY1 请把上一行替换生效

% 导弹沿直线指向原点
u_m   = -M0 / norm(M0);
T_max = norm(M0) / v_m;    %#ok<NASGU>  % 仅信息参考

%% =============== 精度与边界 =================
dt      = 0.01;    % 时间步长（0.005~0.02 常用；越小越准越慢）
nTheta  = 720;     % 每个圆的边缘采样点数（360~720 建议；越大越准越慢）

% 决策变量边界： [theta1(rad),   v_u(m/s),   t_rel(s), t_fuse(s)]
lb = [0,            70,            0,         0.10];
ub = [2*pi,         140,           30,        8.00];

%% =============== SA 超参数 =================
SA.MaxStages    = 180;      % 温度阶段数（外层循环）
SA.L_per_stage  = 220;      % 每阶段马氏链长度（内层采样次数）
SA.alpha        = 0.93;     % 冷却因子（0.90~0.98；越大越慢越稳）
SA.reheat_every = 25;       % 连续无改进阶段阈值（触发轻度重热）
SA.reheat_mult  = 1.4;      % 重热倍率（不会超过初温）
SA.target_acc_hi= 0.60;     % 接受率上阈（步长放大）
SA.target_acc_lo= 0.20;     % 接受率下阈（步长缩小）
SA.step_up      = 1.20;     % 步长放大倍率
SA.step_dn      = 0.80;     % 步长缩小倍率
SA.Tmin         = 1e-4;     % 结束温度阈值（过低即停止）
rng(42);                    % 固定随机种子，便于复现实验

%% =============== 预计算：目标圆柱顶/底“圆周点” =================
rimPts = buildRimPoints(C_t, R_t, H_t, nTheta);  % 2*nTheta×3，仅生成一次，显著提速

%% =============== SA：初始化 =================
% 用匿名函数把常量捕获进来（脚本安全，不会再丢变量）
objFun = @(x) objectiveFunction(x, M0, F0, C_t, u_m, g, v_m, v_s, ...
                                R_s, R_t, H_t, T_eff, dt, rimPts);

% 初始解（可改为启发式）
x0 = lb + (ub - lb).*rand(1,4);

% 自动设定初温：让“典型劣解”有较高接受率 p0≈0.8
p0     = 0.8;
DeltaS = estimateTypicalDelta(x0, 30, lb, ub, objFun);
T0     = max(1e-6, -DeltaS / log(p0));

% 初始步长（各维独立）
sigma0    = 0.10*(ub - lb);
sigma_min = 1e-4*(ub - lb);
sigma_max = 0.25*(ub - lb);

% 初值评估
f0 = objFun(x0);
x  = x0;      fx = f0;
x_best = x;   f_best = fx;

fprintf('改良 SA 启动：T0=%.4g, 预估ΔS=%.4g\n', T0, DeltaS);

%% =============== SA 主循环 ===================
T = T0; sigma = sigma0;
no_improve_stages = 0;

for stage = 1:SA.MaxStages
    n_accept = 0;

    for l = 1:SA.L_per_stage
        % —— 产生邻域候选：高斯扰动 + 边界反射 ——
        x_new = neighbor_gauss_reflect(x, sigma, lb, ub);

        % —— 新解评估 ——
        f_new = objFun(x_new);

        % —— Metropolis 接受准则 ——
        dF = f_new - fx;
        if dF <= 0
            accept = true;   % 更优或等优：必接收
        else
            accept = (rand < exp(-dF / T));  % 劣解：概率接收
        end

        if accept
            x = x_new; fx = f_new;
            n_accept = n_accept + 1;
            if fx < f_best
                x_best = x; f_best = fx;
            end
        end
    end

    % —— 阶段自适应步长（看接受率） ——
    acc_rate = n_accept / SA.L_per_stage;
    if acc_rate > SA.target_acc_hi
        sigma = min(sigma_max, SA.step_up * sigma);
    elseif acc_rate < SA.target_acc_lo
        sigma = max(sigma_min, SA.step_dn * sigma);
    end

    % —— 日志（把负号翻回“遮蔽时长”） ——
    if mod(stage, 10)==0 || stage==1
        fprintf('Stage %3d: T=%.3g, acc=%.2f, best cover=%.3f s\n', ...
                stage, T, acc_rate, -f_best);
    end

    % —— 冷却 ——
    T = max(SA.Tmin, SA.alpha * T);

    % —— 停滞检测与轻度重热 ——
    if n_accept==0
        no_improve_stages = no_improve_stages + 1;
        if no_improve_stages >= SA.reheat_every
            T = min(T0, SA.reheat_mult * T);
            no_improve_stages = 0;
            fprintf('  [reheat] 温度回升至 %.3g 以跳出局部最优\n', T);
        end
    else
        no_improve_stages = 0;
    end

    % —— 终止条件 ——
    if T <= SA.Tmin
        fprintf('温度达到 Tmin=%.2g，提前停止\n', SA.Tmin);
        break;
    end
end

%% =============== 输出与可视化 =================
fprintf('\n=== SA 优化完成（严格口径·圆周采样）===\n');
fprintf('最优参数：\n');
fprintf('  航向角 theta1 = %.4f rad (%.2f°)\n', x_best(1), x_best(1)*180/pi);
fprintf('  FY 速度 v_u    = %.2f m/s\n', x_best(2));
fprintf('  投放延迟 t_rel = %.3f s\n', x_best(3));
fprintf('  引信时间 t_fuse= %.3f s\n', x_best(4));

[fitness_star, P_drop, C_e, t_e] = objectiveFunction(x_best, M0, F0, C_t, u_m, g, v_m, v_s, ...
                                                     R_s, R_t, H_t, T_eff, dt, rimPts);
fprintf('\n关键位置：\n');
fprintf('  投放点 P_drop = (%.2f, %.2f, %.2f) m\n', P_drop);
fprintf('  起爆点 C_e    = (%.2f, %.2f, %.2f) m\n', C_e);
fprintf('  起爆时间 t_e  = %.3f s\n', t_e);
fprintf('\n最大“完全遮蔽”时长：%.3f s\n', -fitness_star);

% 可视化（可选）
visualizeResult(x_best, M0, F0, C_t, R_t, H_t, u_m, g, v_m, v_s, R_s, T_eff, dt, rimPts);

%% =================== 内部函数区 ===================

% 目标函数：返回“负的完全遮蔽时间”（为了最小化）；并回传关键点
function [fitness, P_drop, C_e, t_e] = objectiveFunction(x, M0, F0, C_t, u_m, g, v_m, v_s, ...
                                                         R_s, R_t, H_t, T_eff, dt, rimPts)
    theta1 = x(1); v_u = x(2); t_rel = x(3); t_fuse = x(4);

    % FY 等高直线（方向在 xy 平面）
    dir = [cos(theta1), sin(theta1), 0];

    % 投放点与投放瞬时速度
    P_drop = F0 + v_u * t_rel * dir;
    v_drop = v_u * dir;

    % 起爆点（平抛：水平携带 v_drop，竖直自由落体）
    C_e = P_drop + v_drop * t_fuse;
    C_e(3) = C_e(3) - 0.5 * g * t_fuse^2;

    % 起爆时刻
    t_e = t_rel + t_fuse;

    % 可行性约束（地面以下或时间异常 → 大惩罚）
    if C_e(3) <= 0 || t_e < 0
        fitness = 1e3;
        return;
    end

    % 遮蔽统计（严格口径：顶/底圆周的所有点都被挡才 +dt）
    T_cover = 0;
    t_end = t_e + T_eff;
    for t = t_e:dt:t_end
        % 导弹位置（沿 u_m 匀速 v_m）
        M_pos = M0 + v_m * t * u_m;
        % 云团中心（水平不动，竖直以 v_s 下沉）
        C_cloud = C_e - [0, 0, v_s * (t - t_e)];

        if isOccluded_rims_strict(M_pos, C_cloud, R_s, rimPts)
            T_cover = T_cover + dt;
        end
    end

    fitness = -T_cover;  % 取负以实现“最大化遮蔽时长”
end

% 严格口径判定：仅顶/底“圆周”点；所有点都被挡才算“完全遮蔽”
function occluded = isOccluded_rims_strict(M_pos, C_cloud, R_s, rimPts)
    occluded = true;
    for k = 1:size(rimPts,1)
        P = rimPts(k,:);
        if ~segmentSphereIntersect(M_pos, P, C_cloud, R_s)
            occluded = false;
            return; % 只要有一个圆周点没被挡，就不是“完全遮蔽”
        end
    end
end

% 构建顶/底圆周点（绝对坐标；只算一次）
function rimPts = buildRimPoints(C_t_center, R_t, H_t, nTheta)
    % 把“几何中心”换成“底圆心”
    C_base = [C_t_center(1), C_t_center(2), C_t_center(3) - H_t/2];
    cx = C_base(1); cy = C_base(2); cz = C_base(3);
    z_bot = cz; z_top = cz + H_t;

    thetas = linspace(0, 2*pi, nTheta+1); thetas(end) = []; % 去重
    rimPts = zeros(2*nTheta, 3);
    for i = 1:nTheta
        cth = cos(thetas(i)); sth = sin(thetas(i));
        rimPts(2*i-1,:) = [cx + R_t*cth, cy + R_t*sth, z_bot]; % 底圆边缘
        rimPts(2*i,  :) = [cx + R_t*cth, cy + R_t*sth, z_top]; % 顶圆边缘
    end
end

% 线段-球相交（充要判定，二次方程）
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
        hit = false; return;
    end
    sqrtD = sqrt(D);
    t1 = (-B - sqrtD) / (2*A);
    t2 = (-B + sqrtD) / (2*A);
    t_enter = min(t1, t2);
    t_exit  = max(t1, t2);
    % 与线段相交：存在 t ∈ [0,1]，且 t_enter ≤ 1, t_exit ≥ 0（区间有重叠）
    hit = (t_exit >= 0.0) && (t_enter <= 1.0);
end

% 邻域生成：高斯扰动 + 边界反射（越界就镜像回搜索域）
function x_new = neighbor_gauss_reflect(x, sigma, lb, ub)
    x_new = x + sigma .* randn(size(x));
    for i = 1:numel(x_new)
        % 多次反射以处理“大越界”
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

% 估算典型代价差 ΔS，用于自动设定初温（更稳健）
function DeltaS = estimateTypicalDelta(x_ref, N, lb, ub, fhandle)
    f0 = fhandle(x_ref);
    deltas = zeros(N,1);
    for k = 1:N
        x_try = lb + (ub - lb).*rand(1, numel(x_ref)); % 随机点
        fk = fhandle(x_try);
        deltas(k) = abs(fk - f0);
    end
    DeltaS = median(deltas); % 中位数更稳健
    if ~isfinite(DeltaS) || DeltaS<=0, DeltaS = 1; end
end

% 可视化（轨迹 + 起爆烟幕球）
function visualizeResult(x_star, M0, F0, C_t, R_t, H_t, u_m, g, v_m, v_s, R_s, T_eff, dt, rimPts)
    figure('Position', [100, 100, 1200, 800]);

    % 复算关键点
    [~, P_drop, C_e, t_e] = objectiveFunction(x_star, M0, F0, C_t, u_m, g, v_m, v_s, ...
                                              R_s, R_t, H_t, T_eff, dt, rimPts);

    % 轨迹
    t_missile = 0:0.1:70;
    M_traj = M0 + (v_m .* (t_missile')) .* u_m;

    theta1 = x_star(1); v_u = x_star(2);
    dir = [cos(theta1), sin(theta1), 0];
    t_drone = 0:0.1:x_star(3);
    F_traj = F0 + (v_u .* (t_drone')) .* dir;

    % 绘图
    plot3(M_traj(:,1), M_traj(:,2), M_traj(:,3), 'r-', 'LineWidth', 2); hold on;
    plot3(F_traj(:,1), F_traj(:,2), F_traj(:,3), 'b-', 'LineWidth', 2);

    plot3(M0(1), M0(2), M0(3), 'ro', 'MarkerFaceColor','r', 'DisplayName','M1起点');
    plot3(F0(1), F0(2), F0(3), 'bo', 'MarkerFaceColor','b', 'DisplayName','FY起点');
    plot3(C_t(1), C_t(2), C_t(3), 'go', 'MarkerFaceColor','g', 'DisplayName','真目标中心');
    plot3(P_drop(1), P_drop(2), P_drop(3), 'mo', 'MarkerFaceColor','m', 'DisplayName','投放点');
    plot3(C_e(1), C_e(2), C_e(3), 'co', 'MarkerFaceColor','c', 'DisplayName','起爆点');

    % 起爆烟幕球
    [xs, ys, zs] = sphere(40);
    surf(C_e(1)+R_s*xs, C_e(2)+R_s*ys, C_e(3)+R_s*zs, ...
         'FaceAlpha',0.25,'EdgeColor','none','DisplayName','烟幕球（起爆）');

    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title('FY 对 M1 · 烟幕干扰最优策略（严格口径·仅顶/底圆周）');
    legend({'导弹轨迹','无人机轨迹','M1起点','FY起点','真目标中心','投放点','起爆点','烟幕球'}, ...
           'Location','northeastoutside');
    grid on; axis equal; view(45,30);
end
