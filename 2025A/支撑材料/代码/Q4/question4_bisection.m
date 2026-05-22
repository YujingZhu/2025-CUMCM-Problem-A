%% 问题4：三架无人机协同优化（扫描+二分法快速求解）
% 2025年高教社杯全国大学生数学建模竞赛A题
clear; clc; close all;

%% 全局参数定义
g = 9.81;               % 重力加速度 (m/s²)
v_m = 300;              % 导弹速度 (m/s)
v_s = 3;                % 烟幕下沉速度 (m/s)
R_s = 10;               % 烟幕有效半径 (m)
T_eff = 20;             % 起爆后有效时间 (s)
R_t = 7;                % 真目标底圆半径 (m)
H_t = 10;               % 真目标高度 (m)
dt = 0.05;              % 时间步长 (s) - 加大步长提速

% 位置定义
C_t = [0, 200, 5];      % 真目标几何中心
M0 = [20000, 0, 2000];  % M1初始位置
u_m = -M0 / norm(M0);   % 导弹方向单位向量
T_max = norm(M0) / v_m; % 导弹最大飞行时间

% 三架无人机初始位置
drones_F0 = {
    [17800, 0, 1800],     % FY1
    [12000, 1400, 1400],  % FY2
    [6000, -3000, 700]    % FY3
};

fprintf('开始扫描+二分法快速优化...\n');
tic;

%% 第一步：粗扫描找到每架无人机的最优策略
drone_results = cell(3, 1);

for d = 1:3
    fprintf('\n正在优化 FY%d...\n', d);
    F0_d = drones_F0{d};
    
    % 粗扫描参数范围
    theta_range = linspace(0, 2*pi, 36);        % 航向角：10°步长
    v_u_range = linspace(80, 130, 11);          % 速度：5 m/s步长
    t_rel_range = linspace(2, 25, 12);          % 投放延迟：2s步长
    t_fuse_range = linspace(1, 6, 11);          % 引信时间：0.5s步长
    
    best_score = -inf;
    best_params = [];
    
    % 四重循环粗扫描
    total_combinations = length(theta_range) * length(v_u_range) * length(t_rel_range) * length(t_fuse_range);
    count = 0;
    
    for theta = theta_range
        for v_u = v_u_range
            for t_rel = t_rel_range
                for t_fuse = t_fuse_range
                    count = count + 1;
                    if mod(count, 1000) == 0
                        fprintf('  进度: %.1f%%\n', 100*count/total_combinations);
                    end
                    
                    % 快速评估
                    score = evaluateDroneStrategy([theta, v_u, t_rel, t_fuse], M0, F0_d, C_t, u_m, g, v_m, v_s, R_s, R_t, H_t, T_eff, dt);
                    
                    if score > best_score
                        best_score = score;
                        best_params = [theta, v_u, t_rel, t_fuse];
                    end
                end
            end
        end
    end
    
    fprintf('  FY%d 粗扫描完成，最佳遮蔽时长: %.3f s\n', d, best_score);
    
    %% 第二步：二分法精细优化
    fprintf('  开始二分法精细优化...\n');
    
    % 在最优点附近用二分法细化
    refined_params = binarySearchOptimize(best_params, M0, F0_d, C_t, u_m, g, v_m, v_s, R_s, R_t, H_t, T_eff, dt);
    final_score = evaluateDroneStrategy(refined_params, M0, F0_d, C_t, u_m, g, v_m, v_s, R_s, R_t, H_t, T_eff, dt);
    
    fprintf('  FY%d 最终优化完成，遮蔽时长: %.3f s\n', d, final_score);
    
    % 存储结果
    drone_results{d} = struct('params', refined_params, 'score', final_score, 'F0', F0_d);
end

%% 第三步：计算详细结果并保存
fprintf('\n=== 优化完成，计算最终结果 ===\n');

results = cell(3, 11);
drone_names = {'FY1', 'FY2', 'FY3'};
total_score = 0;

for d = 1:3
    params = drone_results{d}.params;
    F0_d = drone_results{d}.F0;
    
    theta = params(1);
    v_u = params(2);
    t_rel = params(3);
    t_fuse = params(4);
    
    % 计算关键点
    dir = [cos(theta), sin(theta), 0];
    P_drop = F0_d + v_u * t_rel * dir;
    v_drop = v_u * dir;
    C_e = P_drop + v_drop * t_fuse;
    C_e(3) = C_e(3) - 0.5 * g * t_fuse^2;
    
    % 存储结果
    results{d, 1} = drone_names{d};
    results{d, 2} = v_u;                    % 飞行速度
    results{d, 3} = mod(rad2deg(theta), 360); % 飞行方向（0-360°）
    results{d, 4} = t_rel;                  % 投放时刻
    results{d, 5} = t_fuse;                 % 引信时间
    results{d, 6} = P_drop(1);              % 投放点x
    results{d, 7} = P_drop(2);              % 投放点y
    results{d, 8} = P_drop(3);              % 投放点z
    results{d, 9} = C_e(1);                 % 起爆点x
    results{d, 10} = C_e(2);                % 起爆点y
    results{d, 11} = C_e(3);                % 起爆点z
    
    total_score = total_score + drone_results{d}.score;
    
    % 输出详细信息
    fprintf('\n%s 最优参数:\n', drone_names{d});
    fprintf('  航向角: %.1f° | 速度: %.1f m/s | 投放延迟: %.2f s | 引信: %.2f s\n', ...
            mod(rad2deg(theta), 360), v_u, t_rel, t_fuse);
    fprintf('  投放点: (%.1f, %.1f, %.1f) m\n', P_drop);
    fprintf('  起爆点: (%.1f, %.1f, %.1f) m\n', C_e);
    fprintf('  遮蔽时长: %.3f s\n', drone_results{d}.score);
end

fprintf('\n总遮蔽时长: %.3f s\n', total_score);

%% 保存结果到Excel
filename = 'result2.xlsx';
header = {'无人机编号', '飞行速度(m/s)', '飞行方向(°)', '投放时刻(s)', '引信时间(s)', ...
          '投放点x(m)', '投放点y(m)', '投放点z(m)', '起爆点x(m)', '起爆点y(m)', '起爆点z(m)'};

try
    writecell([header; results], filename, 'Sheet', 1, 'Range', 'A1');
    fprintf('\n结果已保存至: %s\n', filename);
catch ME
    fprintf('\nExcel保存失败，显示结果数据:\n');
    disp([header; results]);
end

elapsed_time = toc;
fprintf('总计算时间: %.2f 秒\n', elapsed_time);

%% 可视化结果
visualizeResults(drone_results, drones_F0, M0, C_t, u_m, g, v_m, R_s);

%% =================== 核心函数 ===================

%% 快速评估单架无人机策略
function score = evaluateDroneStrategy(params, M0, F0, C_t, u_m, g, v_m, v_s, R_s, R_t, H_t, T_eff, dt)
    theta = params(1);
    v_u = params(2);
    t_rel = params(3);
    t_fuse = params(4);
    
    % 物理约束检查
    if v_u < 70 || v_u > 140 || t_rel < 0 || t_rel > 30 || t_fuse < 0.1 || t_fuse > 8
        score = 0;
        return;
    end
    
    % 计算起爆点
    dir = [cos(theta), sin(theta), 0];
    P_drop = F0 + v_u * t_rel * dir;
    v_drop = v_u * dir;
    C_e = P_drop + v_drop * t_fuse;
    C_e(3) = C_e(3) - 0.5 * g * t_fuse^2;
    t_e = t_rel + t_fuse;
    
    % 高度约束
    if C_e(3) <= 0 || t_e < 0
        score = 0;
        return;
    end
    
    % 快速遮蔽计算（较大时间步长）
    T_cover = 0;
    t_vec = t_e:dt:(t_e + T_eff);
    
    for t = t_vec
        M_pos = M0 + v_m * t * u_m;
        C_cloud = C_e - [0, 0, v_s * (t - t_e)];
        
        if isOccluded_fast(M_pos, C_t, C_cloud, R_t, H_t, R_s)
            T_cover = T_cover + dt;
        end
    end
    
    score = T_cover;
end

%% 快速遮蔽判定（减少采样点）
function occluded = isOccluded_fast(M_pos, C_t_center, C_cloud, R_t, H_t, R_s)
    C_base = [C_t_center(1), C_t_center(2), C_t_center(3) - H_t/2];
    
    % 减少采样点提速（每30°一个点）
    nTheta = 12;  % 原来360个点，现在24个点
    rimPts = sampleCylinderRims(C_base, R_t, H_t, nTheta);
    
    occluded = true;
    for k = 1:size(rimPts, 1)
        P = rimPts(k, :);
        if ~segmentSphereIntersect(M_pos, P, C_cloud, R_s)
            occluded = false;
            return;
        end
    end
end

%% 高精度二分法精细优化
function best_params = binarySearchOptimize(initial_params, M0, F0, C_t, u_m, g, v_m, v_s, R_s, R_t, H_t, T_eff, dt)
    best_params = initial_params;
    best_score = evaluateDroneStrategy(initial_params, M0, F0, C_t, u_m, g, v_m, v_s, R_s, R_t, H_t, T_eff, dt);
    
    % 更精细的初始搜索范围
    search_ranges = [
        pi/36,   % 航向角 ±5°（比原来的±10°更精细）
        2.5,     % 速度 ±2.5 m/s（比原来的±5 m/s更精细）
        0.5,     % 投放延迟 ±0.5s（比原来的±1s更精细）
        0.25     % 引信时间 ±0.25s（比原来的±0.5s更精细）
    ];
    
    % 增加迭代次数，提升精度
    max_iterations = 12;  % 从8次增加到12次，精度提升到1/3^12 ≈ 1/531441
    
    % 对每个参数进行多轮优化
    for round = 1:3  % 多轮优化，每轮都会进一步细化
        for param_idx = 1:4
            current_params = best_params;
            search_range = search_ranges(param_idx) / round;  % 每轮搜索范围递减
            
            % 黄金分割法搜索（比三分法更精确）
            phi = (1 + sqrt(5)) / 2;  % 黄金比例
            for iter = 1:max_iterations
                left = current_params(param_idx) - search_range;
                right = current_params(param_idx) + search_range;
                
                % 黄金分割点
                m1 = right - (right - left) / phi;
                m2 = left + (right - left) / phi;
                
                % 评估分割点
                params1 = current_params; params1(param_idx) = m1;
                params2 = current_params; params2(param_idx) = m2;
                
                % 使用更精确的评估（减小时间步长）
                score1 = evaluateDroneStrategy_HighPrecision(params1, M0, F0, C_t, u_m, g, v_m, v_s, R_s, R_t, H_t, T_eff);
                score2 = evaluateDroneStrategy_HighPrecision(params2, M0, F0, C_t, u_m, g, v_m, v_s, R_s, R_t, H_t, T_eff);
                
                if score1 > score2
                    current_params(param_idx) = m1;
                    right = m2;
                else
                    current_params(param_idx) = m2;
                    left = m1;
                end
                
                search_range = (right - left) / 2;
                
                % 收敛判断
                if search_range < 1e-6
                    break;
                end
            end
            
            % 更新最优参数
            new_score = evaluateDroneStrategy_HighPrecision(current_params, M0, F0, C_t, u_m, g, v_m, v_s, R_s, R_t, H_t, T_eff);
            if new_score > best_score
                best_params = current_params;
                best_score = new_score;
                fprintf('    参数%d优化: %.6f -> %.6f\n', param_idx, best_score - (new_score - best_score), new_score);
            end
        end
    end
    
    % 最后进行联合优化（同时调整所有参数）
    best_params = jointOptimization(best_params, M0, F0, C_t, u_m, g, v_m, v_s, R_s, R_t, H_t, T_eff);
end

%% 高精度评估函数
function score = evaluateDroneStrategy_HighPrecision(params, M0, F0, C_t, u_m, g, v_m, v_s, R_s, R_t, H_t, T_eff)
    dt_high = 0.01;  % 更小的时间步长，提高精度
    
    theta = params(1);
    v_u = params(2);
    t_rel = params(3);
    t_fuse = params(4);
    
    % 物理约束检查
    if v_u < 70 || v_u > 140 || t_rel < 0 || t_rel > 30 || t_fuse < 0.1 || t_fuse > 8
        score = 0;
        return;
    end
    
    % 计算起爆点
    dir = [cos(theta), sin(theta), 0];
    P_drop = F0 + v_u * t_rel * dir;
    v_drop = v_u * dir;
    C_e = P_drop + v_drop * t_fuse;
    C_e(3) = C_e(3) - 0.5 * g * t_fuse^2;
    t_e = t_rel + t_fuse;
    
    % 高度约束
    if C_e(3) <= 0 || t_e < 0
        score = 0;
        return;
    end
    
    % 高精度遮蔽计算
    T_cover = 0;
    t_vec = t_e:dt_high:(t_e + T_eff);
    
    for t = t_vec
        M_pos = M0 + v_m * t * u_m;
        C_cloud = C_e - [0, 0, v_s * (t - t_e)];
        
        % 使用更精确的遮蔽判定
        if isOccluded_HighPrecision(M_pos, C_t, C_cloud, R_t, H_t, R_s)
            T_cover = T_cover + dt_high;
        end
    end
    
    score = T_cover;
end

%% 高精度遮蔽判定
function occluded = isOccluded_HighPrecision(M_pos, C_t_center, C_cloud, R_t, H_t, R_s)
    C_base = [C_t_center(1), C_t_center(2), C_t_center(3) - H_t/2];
    
    % 增加采样点数量，提高精度
    nTheta = 72;  % 每5°一个点，比快速版本的30°更精细
    rimPts = sampleCylinderRims(C_base, R_t, H_t, nTheta);
    
    occluded = true;
    for k = 1:size(rimPts, 1)
        P = rimPts(k, :);
        if ~segmentSphereIntersect(M_pos, P, C_cloud, R_s)
            occluded = false;
            return;
        end
    end
end

%% 联合优化（多参数同时优化）
function best_params = jointOptimization(initial_params, M0, F0, C_t, u_m, g, v_m, v_s, R_s, R_t, H_t, T_eff)
    best_params = initial_params;
    best_score = evaluateDroneStrategy_HighPrecision(initial_params, M0, F0, C_t, u_m, g, v_m, v_s, R_s, R_t, H_t, T_eff);
    
    % 使用梯度下降的思想进行联合优化
    step_sizes = [pi/180, 0.1, 0.01, 0.01];  % 各参数的步长
    
    for iter = 1:50  % 最大50次迭代
        improved = false;
        
        % 对每个参数方向进行小步长搜索
        for param_idx = 1:4
            for direction = [-1, 1]  % 正负两个方向
                test_params = best_params;
                test_params(param_idx) = test_params(param_idx) + direction * step_sizes(param_idx);
                
                % 边界检查
                if param_idx == 1  % 角度
                    test_params(1) = mod(test_params(1), 2*pi);
                elseif param_idx == 2  % 速度
                    if test_params(2) < 70 || test_params(2) > 140, continue; end
                elseif param_idx == 3  % 投放延迟
                    if test_params(3) < 0 || test_params(3) > 30, continue; end
                elseif param_idx == 4  % 引信时间
                    if test_params(4) < 0.1 || test_params(4) > 8, continue; end
                end
                
                test_score = evaluateDroneStrategy_HighPrecision(test_params, M0, F0, C_t, u_m, g, v_m, v_s, R_s, R_t, H_t, T_eff);
                
                if test_score > best_score
                    best_params = test_params;
                    best_score = test_score;
                    improved = true;
                end
            end
        end
        
        % 如果没有改进，减小步长
        if ~improved
            step_sizes = step_sizes * 0.8;
            if max(step_sizes) < 1e-6
                break;
            end
        end
    end
end
%% 圆柱圆周采样
function rimPts = sampleCylinderRims(C_base, R, H, nTheta)
    cx = C_base(1); cy = C_base(2); cz = C_base(3);
    z_bot = cz; z_top = cz + H;
    
    thetas = linspace(0, 2*pi, nTheta+1);
    thetas(end) = [];
    
    rimPts = zeros(2*nTheta, 3);
    for i = 1:nTheta
        cth = cos(thetas(i)); sth = sin(thetas(i));
        rimPts(2*i-1, :) = [cx + R*cth, cy + R*sth, z_bot];
        rimPts(2*i, :) = [cx + R*cth, cy + R*sth, z_top];
    end
end

%% 线段-球相交检测
function hit = segmentSphereIntersect(M, P, C, r)
    d = P - M;
    f = M - C;
    A = dot(d, d);
    
    if A <= 1e-16
        hit = (dot(f, f) <= r*r + 1e-12);
        return;
    end
    
    B = 2.0 * dot(f, d);
    Q = dot(f, f) - r*r;
    discriminant = B*B - 4*A*Q;
    
    if discriminant <= 0
        hit = false;
        return;
    end
    
    sqrtD = sqrt(discriminant);
    t1 = (-B - sqrtD) / (2*A);
    t2 = (-B + sqrtD) / (2*A);
    t_enter = min(t1, t2);
    t_exit = max(t1, t2);
    
    hit = (t_exit >= 0.0) && (t_enter <= 1.0);
end

%% 结果可视化
function visualizeResults(drone_results, drones_F0, M0, C_t, u_m, g, v_m, R_s)
    figure('Position', [100, 100, 1400, 900], 'Name', '扫描+二分法优化结果');
    
    colors = {'b', 'r', 'm'};
    drone_names = {'FY1', 'FY2', 'FY3'};
    
    % 导弹轨迹
    t_missile = linspace(0, 60, 100);
    M_traj = M0 + v_m * t_missile' * u_m;
    plot3(M_traj(:,1), M_traj(:,2), M_traj(:,3), 'k-', 'LineWidth', 3, 'DisplayName', '导弹M1轨迹');
    hold on;
    
    % 真目标
    [X_cyl, Y_cyl, Z_cyl] = cylinder(7, 20);
    Z_cyl = Z_cyl * 10 + (C_t(3) - 5);
    surf(X_cyl + C_t(1), Y_cyl + C_t(2), Z_cyl, 'FaceAlpha', 0.4, 'FaceColor', 'g', 'EdgeColor', 'none');
    
    % 各无人机轨迹和关键点
    for d = 1:3
        color = colors{d};
        params = drone_results{d}.params;
        F0_d = drones_F0{d};
        
        theta = params(1); v_u = params(2); t_rel = params(3); t_fuse = params(4);
        
        dir = [cos(theta), sin(theta), 0];
        t_drone = linspace(0, t_rel, 30);
        F_traj = F0_d + v_u * t_drone' * dir;
        
        P_drop = F0_d + v_u * t_rel * dir;
        v_drop = v_u * dir;
        C_e = P_drop + v_drop * t_fuse;
        C_e(3) = C_e(3) - 0.5 * g * t_fuse^2;
        
        % 绘制轨迹和点
        plot3(F_traj(:,1), F_traj(:,2), F_traj(:,3), [color '-'], 'LineWidth', 2);
        plot3(F0_d(1), F0_d(2), F0_d(3), [color 'o'], 'MarkerSize', 12, 'MarkerFaceColor', color);
        plot3(P_drop(1), P_drop(2), P_drop(3), [color 's'], 'MarkerSize', 12, 'MarkerFaceColor', color);
        plot3(C_e(1), C_e(2), C_e(3), [color '^'], 'MarkerSize', 12, 'MarkerFaceColor', color);
        
        % 烟幕球
        [xs, ys, zs] = sphere(15);
        surf(C_e(1) + R_s*xs, C_e(2) + R_s*ys, C_e(3) + R_s*zs, ...
             'FaceAlpha', 0.3, 'FaceColor', color, 'EdgeColor', 'none');
    end
    
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title('三机协同烟幕干扰策略（扫描+二分法优化）', 'FontSize', 16, 'FontWeight', 'bold');
    grid on; axis equal; view(45, 30);
    legend('导弹轨迹', '真目标', 'Location', 'best');
end
