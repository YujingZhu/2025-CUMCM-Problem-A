clc; clear; close all;

%% ------------------ 常量设置 ------------------
R_smoke = 10;               
N = 100;                    
line_width = 1.5;  
g_fall = 9.8;                % 平抛下落加速度

%% ------------------ 导弹参数 ------------------
m0 = [20000, 0, 2000];       
target = [0, 0, 0];          
v_m = 300;                     
T_m = norm(target - m0) / v_m;
t_m = linspace(0, T_m, N);
missile_traj = m0 + (target - m0) / T_m .* t_m';

%% ------------------ 无人机参数（FY1-FY3） ------------------
drones = {...
    struct('name','FY1','pos',[17800,0,1800],'theta',7.19,'v',131.89,'drop',[17800,0,1800],'explode',[17996.68,12.2,1797.33]),...
    struct('name','FY2','pos',[12000,1400,1400],'theta',296.18,'v',113.18,'drop',[12417.87, 550.18, 1400],'explode',[12671.06,35.29,1273.93]),...
    struct('name','FY3','pos',[6000,-3000,700],'theta',75.98,'v',104.49,'drop',[6572.52, -115.23, 700],'explode',[6767.08,71.17,683.42])...
};

%% ------------------ 马卡龙色系配色定义 ------------------
% 使用马卡龙浅色系
col_missile = [1.0, 0.75, 0.8];        % 马卡龙粉色 - 导弹轨迹
col_drone = [0.75, 0.85, 1.0];          % 马卡龙蓝色 - 无人机
col_drop = [1.0, 0.95, 0.7];           % 马卡龙黄色 - 投放轨迹  
col_explode = [0.9, 0.7, 0.8];        % 马卡龙紫粉色 - 爆炸轨迹
col_smoke = [0.85, 0.85, 0.85];          % 马卡龙灰色 - 烟幕
col_target_fake = [0.6, 0.6, 0.6];    % 中灰色 - 假目标
col_target_real = [0.8, 0.95, 0.8];    % 马卡龙绿色 - 真目标

arrow_scale_m = 800;  
arrow_scale_u = 400;  
arrow_scale_drop = 50; 

%% ------------------ 绘图 ------------------
figure; 
set(gcf, 'Position', [100, 100, 1000, 800]); % 增大图形窗口
set(gcf, 'Color', 'white'); % 白色背景
hold on; grid on; axis equal;

% 设置坐标轴样式
xlabel('X (m)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Y (m)', 'FontSize', 14, 'FontWeight', 'bold');
zlabel('Z (m)', 'FontSize', 14, 'FontWeight', 'bold');
set(gca, 'FontSize', 12);
set(gca, 'GridAlpha', 0.3); % 调整网格透明度
set(gca, 'LineWidth', 1.2); % 调整坐标轴线宽

%% ====== 1. 导弹轨迹 ======
plot3(missile_traj(:,1), missile_traj(:,2), missile_traj(:,3), '-', 'Color', [0.8, 0.4, 0.5], 'LineWidth', 3);
scatter3(m0(1), m0(2), m0(3), 120, 'o', 'filled', 'MarkerFaceColor', col_missile);
dir_m = (target - m0)/norm(target - m0);
quiver3(m0(1), m0(2), m0(3), dir_m(1)*arrow_scale_m, dir_m(2)*arrow_scale_m, dir_m(3)*arrow_scale_m, ...
    0, 'Color', [0.8, 0.4, 0.5], 'LineWidth', 2.5, 'MaxHeadSize', 4);

% 导弹起点标注 - 整齐排列
text(m0(1)+500, m0(2), m0(3)+500, '导弹起点', 'Color', [0.8, 0.4, 0.5], 'FontSize', 13, 'FontWeight', 'bold');
text(m0(1)+500, m0(2), m0(3)+350, sprintf('X: %.0f', m0(1)), 'Color', [0.8, 0.4, 0.5], 'FontSize', 12);
text(m0(1)+500, m0(2), m0(3)+200, sprintf('Y: %.0f', m0(2)), 'Color', [0.8, 0.4, 0.5], 'FontSize', 12);
text(m0(1)+500, m0(2), m0(3)+50, sprintf('Z: %.0f', m0(3)), 'Color', [0.8, 0.4, 0.5], 'FontSize', 12);

% 导弹末端方向箭头
arrow_tail_m = missile_traj(end-1,:);
arrow_head_m = missile_traj(end,:);
dir_m_traj = arrow_head_m - arrow_tail_m;
quiver3(arrow_tail_m(1), arrow_tail_m(2), arrow_tail_m(3), ...
        dir_m_traj(1), dir_m_traj(2), dir_m_traj(3), ...
        0, 'Color', [0.8, 0.4, 0.5], 'LineWidth', 2.5, 'MaxHeadSize', 2);

%% ====== 2. 无人机轨迹与烟幕 ======
% 预设整齐的文本偏移量
text_base_x = 1000;  % 统一的X偏移基准
text_base_y = 1000;  % 统一的Y偏移基准
text_base_z = 400;   % 统一的Z偏移基准

for k = 1:length(drones)
    dr = drones{k};
    
    % 无人机轨迹（直线）
    t_u = linspace(0, 50, N);
    theta_rad = deg2rad(dr.theta);
    u_x = dr.pos(1) + dr.v * cos(theta_rad) * t_u;
    u_y = dr.pos(2) + dr.v * sin(theta_rad) * t_u;
    u_z = ones(size(t_u)) * dr.pos(3);
    drone_traj = [u_x', u_y', u_z'];
    plot3(drone_traj(:,1), drone_traj(:,2), drone_traj(:,3), '-', 'Color', [0.4, 0.6, 0.8], 'LineWidth', 3);
    scatter3(dr.pos(1), dr.pos(2), dr.pos(3), 120, 's', 'filled', 'MarkerFaceColor', col_drone);

    % 无人机标注 - 整齐排列，每个无人机使用固定偏移
    if k == 1  % FY1
        offset_x = text_base_x;
        offset_y = text_base_y;
    elseif k == 2  % FY2
        offset_x = -text_base_x;
        offset_y = text_base_y;
    else  % FY3
        offset_x = text_base_x;
        offset_y = -text_base_y;
    end
    
    text(dr.pos(1)+offset_x, dr.pos(2)+offset_y, dr.pos(3)+text_base_z, dr.name, 'Color', [0.4, 0.6, 0.8], 'FontSize', 14, 'FontWeight', 'bold');
    text(dr.pos(1)+offset_x, dr.pos(2)+offset_y, dr.pos(3)+text_base_z-120, sprintf('X: %.0f', dr.pos(1)), 'Color', [0.4, 0.6, 0.8], 'FontSize', 12);
    text(dr.pos(1)+offset_x, dr.pos(2)+offset_y, dr.pos(3)+text_base_z-240, sprintf('Y: %.0f', dr.pos(2)), 'Color', [0.4, 0.6, 0.8], 'FontSize', 12);
    text(dr.pos(1)+offset_x, dr.pos(2)+offset_y, dr.pos(3)+text_base_z-360, sprintf('Z: %.0f', dr.pos(3)), 'Color', [0.4, 0.6, 0.8], 'FontSize', 12);

    % 无人机起点箭头
    dir_u = [cos(theta_rad), sin(theta_rad), 0];
    quiver3(dr.pos(1), dr.pos(2), dr.pos(3), dir_u(1)*arrow_scale_u, dir_u(2)*arrow_scale_u, 0, 0, ...
        'Color', [0.4, 0.6, 0.8], 'LineWidth', 2.5, 'MaxHeadSize', 4);

    % 无人机末端箭头
    arrow_tail_u = drone_traj(end-1,:);
    arrow_head_u = drone_traj(end,:);
    dir_u_traj = arrow_head_u - arrow_tail_u;
    quiver3(arrow_tail_u(1), arrow_tail_u(2), arrow_tail_u(3), ...
            dir_u_traj(1), dir_u_traj(2), dir_u_traj(3), ...
            0, 'Color', [0.4, 0.6, 0.8], 'LineWidth', 2.5, 'MaxHeadSize', 2);

    %% -- 2.1 平抛轨迹（投放点到起爆点） --
    t_fall = sqrt(2*(dr.drop(3)-dr.explode(3))/g_fall);  
    t_line = linspace(0, t_fall, N);

    v_x = dr.v * cos(theta_rad);
    v_y = dr.v * sin(theta_rad);
    x_line = dr.drop(1) + v_x * t_line;
    y_line = dr.drop(2) + v_y * t_line;
    z_line = dr.drop(3) - 0.5 * g_fall * t_line.^2;
    z_line(z_line < dr.explode(3)) = dr.explode(3);
    plot3(x_line, y_line, z_line, '--', 'Color', [0.8, 0.7, 0.3], 'LineWidth', 2.5);

    %% -- 2.2 起爆后匀速下落 --
    t_down = linspace(0, dr.explode(3)/3, N);
    x_down = ones(size(t_down))*dr.explode(1);
    y_down = ones(size(t_down))*dr.explode(2);
    z_down = dr.explode(3) - 3*t_down;
    z_down(z_down<0)=0;
    plot3(x_down, y_down, z_down, ':', 'Color', [0.7, 0.5, 0.6], 'LineWidth', 3);

    % 匀速下落末端箭头
    arrow_length_d = 300;
    quiver3(dr.explode(1), dr.explode(2), dr.explode(3), ...
        0, 0, -arrow_length_d, ...
        0, 'Color', [0.7, 0.5, 0.6], 'LineWidth', 3.5, 'MaxHeadSize', 8);

    %% -- 投放点 & 起爆点 & 烟幕球体 --
    scatter3(dr.drop(1), dr.drop(2), dr.drop(3), 140, 'o', 'filled', 'MarkerFaceColor', col_drop);
    scatter3(dr.explode(1), dr.explode(2), dr.explode(3), 140, 'o', 'filled', 'MarkerFaceColor', col_explode);

    % 投放点标注 - 整齐排列
    drop_offset_x = offset_x * 0.6;
    drop_offset_y = offset_y * 0.6;
    text(dr.drop(1)+drop_offset_x, dr.drop(2)+drop_offset_y, dr.drop(3)+200, '投放点', 'Color', [0.8, 0.7, 0.3], 'FontSize', 12, 'FontWeight', 'bold');
    text(dr.drop(1)+drop_offset_x, dr.drop(2)+drop_offset_y, dr.drop(3)+80, sprintf('(%.0f,%.0f,%.0f)', dr.drop), 'Color', [0.8, 0.7, 0.3], 'FontSize', 11);

    % 起爆点标注 - 整齐排列
    explode_offset_x = offset_x * 0.8;
    explode_offset_y = offset_y * 0.8;
    text(dr.explode(1)+explode_offset_x, dr.explode(2)+explode_offset_y, dr.explode(3)+200, '起爆点', 'Color', [0.7, 0.5, 0.6], 'FontSize', 12, 'FontWeight', 'bold');
    text(dr.explode(1)+explode_offset_x, dr.explode(2)+explode_offset_y, dr.explode(3)+80, sprintf('(%.0f,%.0f,%.0f)', dr.explode), 'Color', [0.7, 0.5, 0.6], 'FontSize', 11);

    [sx, sy, sz] = sphere(30);
    surf(R_smoke*sx + dr.explode(1), R_smoke*sy + dr.explode(2), R_smoke*sz + dr.explode(3), ...
        'FaceAlpha', 0.4, 'EdgeColor', 'none', 'FaceColor', col_smoke);
end

%% ====== 3. 假目标与真目标 ======
scatter3(0, 0, 0, 140, 'filled', 'MarkerFaceColor', col_target_fake);
% 假目标标注 - 整齐排列
text(-800, -800, 300, '假目标', 'Color', [0.6, 0.6, 0.6], 'FontSize', 13, 'FontWeight', 'bold');
text(-800, -800, 180, 'X: 0', 'Color', [0.6, 0.6, 0.6], 'FontSize', 12);
text(-800, -800, 60, 'Y: 0', 'Color', [0.6, 0.6, 0.6], 'FontSize', 12);
text(-800, -800, -60, 'Z: 0', 'Color', [0.6, 0.6, 0.6], 'FontSize', 12);

scatter3(0, 200, 0, 140, 'filled', 'MarkerFaceColor', col_target_real);
% 真目标标注 - 整齐排列
text(-800, 1000, 300, '真目标', 'Color', [0.5, 0.7, 0.5], 'FontSize', 13, 'FontWeight', 'bold');
text(-800, 1000, 180, 'X: 0', 'Color', [0.5, 0.7, 0.5], 'FontSize', 12);
text(-800, 1000, 60, 'Y: 200', 'Color', [0.5, 0.7, 0.5], 'FontSize', 12);
text(-800, 1000, -60, 'Z: 0', 'Color', [0.5, 0.7, 0.5], 'FontSize', 12);

%% ====== 4. 添加图例 ======
legend_elements = {...
    plot3(NaN, NaN, NaN, '-', 'Color', [0.8, 0.4, 0.5], 'LineWidth', 3), ...
    plot3(NaN, NaN, NaN, '-', 'Color', [0.4, 0.6, 0.8], 'LineWidth', 3), ...
    plot3(NaN, NaN, NaN, '--', 'Color', [0.8, 0.7, 0.3], 'LineWidth', 2.5), ...
    plot3(NaN, NaN, NaN, ':', 'Color', [0.7, 0.5, 0.6], 'LineWidth', 3), ...
    scatter3(NaN, NaN, NaN, 60, 'filled', 'MarkerFaceColor', col_smoke), ...
    scatter3(NaN, NaN, NaN, 60, 'filled', 'MarkerFaceColor', col_target_fake), ...
    scatter3(NaN, NaN, NaN, 60, 'filled', 'MarkerFaceColor', col_target_real)...
};

legend([legend_elements{:}], {'导弹轨迹', '无人机轨迹', '平抛轨迹', '爆炸下降', '烟幕区域', '假目标', '真目标'}, ...
    'Location', 'northeast', 'FontSize', 11, 'Box', 'on');

%% ====== 5. 调整视图 ======
axis equal;
view(35,25);
grid on;

% 设置坐标轴范围以优化显示效果
all_x = [missile_traj(:,1); cell2mat({drones{:}.pos}')'];
all_y = [missile_traj(:,2); cell2mat({drones{:}.pos}')'];
xlim([min(all_x)-2000, max(all_x)+2000]);
ylim([min(all_y)-2000, max(all_y)+2000]);

title('无人机烟幕干扰导弹轨迹仿真', 'FontSize', 16, 'FontWeight', 'bold');
