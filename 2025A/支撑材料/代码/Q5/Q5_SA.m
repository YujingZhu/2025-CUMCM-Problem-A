% -*- coding: utf-8 -*-
% FY1 对 M1/M2/M3 三导弹 — 模拟退火（SA）精简版
% 仅保留：多导弹目标 + SA 优化；所有函数均以 end 结尾，避免“函数未以 end 结束”的报错。

clear; clc;
rng(2025,'twister');   % 固定随机种子，便于复现

% ===================== 全局场景参数（global SC） =====================
global SC;
SC.target_x      = 0.0;
SC.target_y      = 200.0;
SC.target_z0     = 0.0;
SC.cyl_radius    = 7.0;
SC.cyl_height    = 10.0;

SC.smoke_radius  = 10.0;   % 烟幕球半径
SC.smoke_v_down  = 3.0;    % 烟幕下沉速度
SC.smoke_lifetime= 20.0;   % 烟幕有效时间

SC.g             = 9.8;

SC.decoy_pos     = [0.0, 0.0, 0.0];
SC.missile_pos_list = [ ...
    20000,    0, 2000;  % M1
    19000,  600, 2100;  % M2
    18000, -600, 1900]; % M3

SC.missile_speed = 300.0;

% FY1 位置（如需改回题面 FY1，请设为 [17800, 0, 1800]）
SC.fy1_pos0      = [11000.0, 2000.0, 1800.0];

SC.dt            = 0.01;
SC.refine_times  = 10;      % 边沿二分精度
SC.max_det_time  = 35.0;    % 起爆时间上限
SC.max_throw_time= 67.0;
SC.min_throw_interval = 1.0;

% ===================== 入口：三导弹 + SA =====================
run_q3_progress_multi();     % 同时对 M1/M2/M3，总时长最大化（SA）

% =======================================================================
%                           函数区（仅保留必要部分）
% =======================================================================

function run_q3_progress_multi()
    % ====== 模拟退火参数（可按耗时调） ======
    starts = 5;            % 多次随机重启
    iters  = 2500;         % 每次退火迭代次数
    T0     = 1.0;          % 初始温度
    alpha  = 0.996;        % 冷却系数
    sigma  = [deg2rad(8), 12, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2];  % 各维扰动幅度

    fprintf('[SA][三导弹] 搜索开始...\n');

    best_overall_x = []; best_overall_J = -inf;
    for s = 1:starts
        x0 = de_rand_vec();  % 8维随机起点：theta,v,tr1,tau1,tr2,tau2,tr3,tau3
        [x_best, j_best] = sa_optimize(@eval_one_multi, x0, iters, T0, alpha, sigma, '[SA][三导弹]');
        if j_best > best_overall_J
            best_overall_J = j_best;
            best_overall_x = x_best;
        end
        fprintf('[SA][三导弹] restart %d/%d done, best J_total=%.6f (so far)\n', s, starts, best_overall_J);
    end

    x_ref = best_overall_x;

    th   = x_ref(1); v = x_ref(2);
    tr1  = x_ref(3); tau1 = x_ref(4);
    tr2  = x_ref(5); tau2 = x_ref(6);
    tr3  = x_ref(7); tau3 = x_ref(8);

    [J_total, pack] = objective_fn_multi(th, v, [tr1,tr2,tr3], [tau1,tau2,tau3]);

    % ------------------ 输出结果 ------------------
    fprintf('\n===== FY1 对 M1/M2/M3 总严格遮蔽时长最大化（SA） =====\n');
    fprintf('无人机方向(°)=%.6f, 无人机速度=%.6f m/s\n', rad2deg(th), v);

    names = {'M1','M2','M3'};
    for m=1:3
        bombs_m = pack.bombs_all{m};
        T_cov_m = pack.T_cov_all{m};
        fprintf('\n--- %s 严格遮蔽 ---\n', names{m});
        header = [ ...
            '弹序号  投放点x(m)     投放点y(m)     投放点z(m)     ', ...
            '起爆点x(m)     起爆点y(m)     起爆点z(m)     有效干扰时段(s)'];
        fprintf('%s\n', header);
        fprintf('%s\n', repmat('-',1,length(header)));
        for idx=1:3
            part = participate_intervals(bombs_m(idx), T_cov_m);
            part_str = format_intervals(part);
            r_launch = bombs_m(idx).r_launch; r_det = bombs_m(idx).r_det;
            fprintf('%6d  %12.6f  %12.6f  %12.6f  %12.6f  %12.6f  %12.6f  %s\n', ...
                idx, r_launch(1), r_launch(2), r_launch(3), ...
                r_det(1), r_det(2), r_det(3), part_str);
        end
        cov_str = format_intervals(T_cov_m);
        fprintf('整体严格遮蔽时间段（%s）：%s\n', names{m}, cov_str);
        fprintf('覆盖时间 J_%s=%.6f s\n', names{m}, union_len(T_cov_m));
    end

    fprintf('\n>>> 三导弹总覆盖时间 J_total = %.6f s\n', J_total);
end

% ---------- SA 通用优化器 ----------
function [x_best, j_best] = sa_optimize(eval_fn, x0, iters, T0, alpha, sigma, tag)
    x_cur = enforce_bounds(x0);
    j_cur = eval_fn(x_cur);
    x_best = x_cur; j_best = j_cur;

    T = T0;
    print_every = max(50, floor(iters/20));

    for k = 1:iters
        % 产生邻域并裁剪到边界
        y = x_cur + randn(1,8).*sigma;
        y = enforce_bounds(y);
        j_new = eval_fn(y);

        % Metropolis 接受准则（最大化）
        if j_new >= j_cur || rand < exp( (j_new - j_cur) / max(1e-12,T) )
            x_cur = y; j_cur = j_new;
            if j_new > j_best
                x_best = y; j_best = j_new;
            end
        end

        % 冷却
        T = T * alpha;

        if mod(k, print_every) == 0
            fprintf('%s iter=%d/%d, curr J=%.6f, best J=%.6f, T=%.4g\n', ...
                tag, k, iters, j_cur, j_best, T);
        end
    end
end

function x = enforce_bounds(x)
    global SC
    x(1) = mod(x(1), 2*pi);               % theta
    x(2) = min(max(x(2),70),140);         % v
    x(3) = min(max(x(3),0),SC.max_throw_time);
    x(5) = min(max(x(5),0),SC.max_throw_time);
    x(7) = min(max(x(7),0),SC.max_throw_time);
    x(4) = min(max(x(4),0),8);            % tau1
    x(6) = min(max(x(6),0),8);            % tau2
    x(8) = min(max(x(8),0),8);            % tau3

    % 简单满足投掷间隔（精确投影仍交由 objective 内部处理）
    trels = sort([x(3), x(5), x(7)]);
    if trels(2) < trels(1) + SC.min_throw_interval
        trels(2) = min(SC.max_throw_time, trels(1) + SC.min_throw_interval);
    end
    if trels(3) < trels(2) + SC.min_throw_interval
        trels(3) = min(SC.max_throw_time, trels(2) + SC.min_throw_interval);
    end
    x(3)=trels(1); x(5)=trels(2); x(7)=trels(3);
end

% ---------- 多导弹：个体评价（目标即三导弹总遮蔽时长） ----------
function J = eval_one_multi(x)
    th=x(1); v=x(2);
    tr1=x(3); tau1=x(4);
    tr2=x(5); tau2=x(6);
    tr3=x(7); tau3=x(8);
    [J,~] = objective_fn_multi(th, v, [tr1,tr2,tr3], [tau1,tau2,tau3]);
end

% ---------- 多导弹：目标函数 ----------
function [J_total, pack] = objective_fn_multi(theta, v, trels, taus)
    [theta, v, trels, taus] = project_params(theta, v, trels, taus);
    global SC
    pos_list = SC.missile_pos_list;

    bombs_all = cell(1,3);
    Tcov_all  = cell(1,3);
    Jsum = 0.0;

    for m = 1:3
        mpos0 = pos_list(m,:);                        % 本枚导弹初始位置
        bombs_m = repmat(struct( ...
            'r_launch', [], 'r_det', [], 't_det', [], ...
            'per_point', [], 'own', []), 1, 3);
        for k=1:3
            b = bomb_intervals_for_missile(theta, v, trels(k), taus(k), mpos0);
            if isempty(b)
                J_total = -1e9; pack = [];
                return;
            end
            bombs_m(k) = b;
        end
        [T_cov_m, Jm] = strict_union_3(bombs_m(1), bombs_m(2), bombs_m(3));
        bombs_all{m} = bombs_m;
        Tcov_all{m}  = T_cov_m;
        Jsum = Jsum + Jm;
    end

    J_total      = Jsum;
    pack.theta   = theta; 
    pack.v       = v; 
    pack.trels   = trels; 
    pack.taus    = taus;
    pack.bombs_all = bombs_all;
    pack.T_cov_all = Tcov_all;
end

% ---------- 多导弹：针对指定导弹的“单点覆盖时间” ----------
function intervals = intervals_for_point_missile(theta, v, t_rel, tau, idx, mpos0)
    global SC
    [~, r_det, t_det] = detonation_point(theta, v, t_rel, tau);
    if ~(0 <= t_det && t_det <= SC.max_det_time)
        intervals = zeros(0,2); return;
    end
    ts = t_det:SC.dt:(t_det + SC.smoke_lifetime + 1e-12);
    if isempty(ts), intervals = zeros(0,2); return; end

    function s = state(t)
        c = smoke_center(t, r_det, t_det);
        if isempty(c)
            s = false; return;
        end
        pts = two_sides();
        s = segment_sphere_intersect(missile_position_with_pos(t, mpos0), pts(idx,:), c, SC.smoke_radius);
    end

    out = zeros(0,2);
    prev_s = state(ts(1));
    if prev_s
        out(end+1,:) = [ts(1), NaN]; %#ok<AGROW>
    end
    for k=2:numel(ts)
        t = ts(k); s = state(t);
        if s ~= prev_s
            lo = ts(k-1); hi = t; slo = prev_s;
            for rr=1:SC.refine_times
                mid = 0.5*(lo+hi);
                sm = state(mid);
                if sm == slo
                    lo = mid;
                else
                    hi = mid;
                end
            end
            tsw = 0.5*(lo+hi);
            if ~prev_s && s
                out(end+1,:) = [tsw, NaN]; %#ok<AGROW>
            elseif prev_s && ~s
                if ~isempty(out) && isnan(out(end,2))
                    out(end,2) = tsw;
                end
            end
        end
        prev_s = s;
    end
    if prev_s && ~isempty(out) && isnan(out(end,2))
        out(end,2) = ts(end);
    end
    good = out(:,2) - out(:,1) > 1e-9;
    intervals = merge_intervals(out(good,:));
end

% ---------- 多导弹：针对指定导弹的“单弹区间集合” ----------
function b = bomb_intervals_for_missile(theta, v, t_rel, tau, mpos0)
    per_point = cell(1,6);
    for i=1:6
        per_point{i} = intervals_for_point_missile(theta, v, t_rel, tau, i, mpos0);
    end
    own_union = merge_intervals(vertcat(per_point{:}));
    [r_launch, r_det, t_det] = detonation_point(theta, v, t_rel, tau);
    b.r_launch = r_launch;
    b.r_det    = r_det;
    b.t_det    = t_det;
    b.per_point= per_point;
    b.own      = own_union;
end

% ---------- 多导弹：带指定初始位置的导弹轨迹 ----------
function p = missile_position_with_pos(t, mpos0)
    global SC
    dir = unit_v(SC.decoy_pos - mpos0);
    p = mpos0 + SC.missile_speed * dir * t;
end

% ===================== 约束投影（保持原逻辑） =====================
function [theta, v, trels, taus] = project_params(theta, v, trels, taus)
    global SC
    theta = mod(theta, 2*pi);
    v = min(max(v,70.0),140.0);
    trels = min(max(trels,0.0), SC.max_throw_time);
    trels = sort(trels);
    for i=2:3
        if trels(i) < trels(i-1) + SC.min_throw_interval
            trels(i) = min(SC.max_throw_time, trels(i-1) + SC.min_throw_interval);
        end
    end
    taus = max(taus, 0.0);
    for k=1:3
        t_det = trels(k) + taus(k);
        if t_det > SC.max_det_time
            over = t_det - SC.max_det_time;
            dec_tau = min(taus(k), over);
            taus(k) = taus(k) - dec_tau;
            over = over - dec_tau;
            if over > 1e-9
                trels(k) = max(0.0, trels(k) - over);
            end
        end
    end
end

% ===================== 基础几何与运动/遮蔽判定 =====================
function v = unit_v(x)
    n = sqrt(sum(x.^2));
    if n==0, v = x; else, v = x./n; end
end

function p = fy1_position(t, theta, v)
    global SC
    dir_xy = [cos(theta), sin(theta), 0.0];
    p = SC.fy1_pos0 + v * dir_xy * t;
end

function [r_launch, r_detonation, t_det] = detonation_point(theta, v, t_rel, tau)
    global SC
    r_launch = fy1_position(t_rel, theta, v);
    accel = [0.0, 0.0, -SC.g];
    r_detonation = r_launch + [v*cos(theta), v*sin(theta), 0.0]*tau + 0.5*accel*(tau^2);
    t_det = t_rel + tau;
end

function c = smoke_center(t, r_d, t_det)
    global SC
    if t < t_det, c = []; return; end
    dt = t - t_det;
    if dt > SC.smoke_lifetime, c = []; return; end
    c = r_d + [0.0, 0.0, -SC.smoke_v_down*dt];
end

function pts = two_sides()
    global SC
    num_radial = 8; num_height = 2;
    TX = SC.target_x; TY = SC.target_y; Z0 = SC.target_z0; R = SC.cyl_radius; H = SC.cyl_height;
    zs = linspace(Z0, Z0+H, num_height);
    pts = zeros(num_radial*num_height + 2, 3);
    idx = 1;
    for z = zs
        for i=0:num_radial-1
            th = 2*pi*i/num_radial;
            x = TX + R*cos(th);
            y = TY + R*sin(th);
            pts(idx,:) = [x,y,z]; idx = idx+1;
        end
    end
    pts(idx,:) = [TX, TY, Z0+H]; idx=idx+1;   % 顶部中心
    pts(idx,:) = [TX, TY, Z0];                % 底部中心
end

function tf = segment_sphere_intersect(p0, p1, c, R)
    d = p1 - p0; f = p0 - c;
    a = dot(d,d); b = 2*dot(f,d); c0 = dot(f,f) - R^2;
    disc = b*b - 4*a*c0;
    if disc < 0, tf = false; return; end
    sdc = sqrt(max(0.0, disc));
    s1 = (-b - sdc)/(2*a);
    s2 = (-b + sdc)/(2*a);
    tf = (0.0 <= s1 && s1 <= 1.0) || (0.0 <= s2 && s2 <= 1.0);
end

% ===================== 时间区间操作 =====================
function intervals = merge_intervals(intervals)
    if isempty(intervals), intervals = zeros(0,2); return; end
    intervals = sortrows(intervals,1);
    merged = intervals(1,:);
    for i=2:size(intervals,1)
        a = intervals(i,1); b = intervals(i,2);
        la = merged(end,1); lb = merged(end,2);
        if a <= lb + 1e-9
            merged(end,2) = max(lb, b);
        else
            merged(end+1,:) = [a,b]; %#ok<AGROW>
        end
    end
    intervals = merged;
end

function out = intersect_two(A,B)
    i=1; j=1; out = zeros(0,2);
    while i<=size(A,1) && j<=size(B,1)
        a1=A(i,1); b1=A(i,2);
        a2=B(j,1); b2=B(j,2);
        a = max(a1,a2); b = min(b1,b2);
        if a < b
            out(end+1,:) = [a,b]; %#ok<AGROW>
        end
        if b1 < b2
            i=i+1;
        else
            j=j+1;
        end
    end
end

function out = intersect_many(list_of_intervals)
    out = list_of_intervals{1};
    for k=2:numel(list_of_intervals)
        out = intersect_two(out, list_of_intervals{k});
        if isempty(out), break; end
    end
end

function L = union_len(U)
    if isempty(U), L = 0.0; else, L = sum(U(:,2)-U(:,1)); end
end

% ===================== 工具 & 初始化 =====================
function s = format_intervals(U)
    if isempty(U), s = '[]'; return; end
    parts = strings(size(U,1),1);
    for i=1:size(U,1)
        parts(i) = sprintf('[%.3f,%.3f]', U(i,1), U(i,2));
    end
    s = "[" + strjoin(parts', ', ') + "]";
    s = char(s);
end

function x = de_rand_vec()
    % 仅作为8维随机起点生成器使用（名字沿用，非DE）
    global SC
    theta = 2*pi*rand();
    v     = 70 + (140-70)*rand();
    trels = sort(SC.max_throw_time*rand(1,3));
    if trels(2) < trels(1) + SC.min_throw_interval
        trels(2) = min(SC.max_throw_time, trels(1) + SC.min_throw_interval);
    end
    if trels(3) < trels(2) + SC.min_throw_interval
        trels(3) = min(SC.max_throw_time, trels(2) + SC.min_throw_interval);
    end
    taus  = 8*rand(1,3);
    x = [theta, v, trels(1), taus(1), trels(2), taus(2), trels(3), taus(3)];
end


function [T_cov, J] = strict_union_3(B1,B2,B3)
    % 对圆柱体6个采样点分别合并三枚烟幕弹的覆盖区间，
    % 然后取6个点的时间交集，得到“严格遮蔽”时间段。
    unions = cell(1,6);
    for i = 1:6
        Ui = [B1.per_point{i}; B2.per_point{i}; B3.per_point{i}];
        if isempty(Ui)
            unions{i} = zeros(0,2);
        else
            unions{i} = merge_intervals(Ui);
        end
    end
    T_cov = intersect_many(unions);
    J = union_len(T_cov);
end

function part = participate_intervals(B, T_cov)
    % 计算某一枚烟幕弹在整体严格遮蔽段内的“参与”区间
    if isempty(T_cov) || isempty(B.own)
        part = zeros(0,2);
    else
        part = intersect_two(T_cov, B.own);
    end
end
