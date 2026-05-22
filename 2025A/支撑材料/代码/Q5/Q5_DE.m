% -*- coding: utf-8 -*-
% Q3：FY1 投放三枚干扰弹拦截 M1 导弹 —— MATLAB R2024a 可运行脚本（global 版）
% 直接运行本脚本即可开始优化，并在命令行显示进度与结果。

clear; clc;
rng(2025,'twister');   % 与 Python 版本保持一致的随机种子

% ===================== 场景与参数定义（改为 global SC） =====================
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
SC.fy1_pos0      = [13000.0, -2000.0, 1300.0];

SC.dt            = 0.01;
SC.refine_times  = 10;      % 边沿二分精度
SC.max_det_time  = 35.0;    % 起爆时间上限
SC.max_throw_time= 67.0;
SC.min_throw_interval = 1.0;

% ===================== 主程序 =====================
% run_q3_progress();   % 入口
% run_q3_progress();          % 仅优化对 M1 的情形（保留原功能）
run_q3_progress_multi();       % 新增：同时对 M1/M2/M3，总时长最大化

% =======================================================================
%                           本脚本所需函数
% =======================================================================

function run_q3_progress()
    % 差分进化参数（与 Python 版本一致）
    pop = 56; gens = 50; F = 0.7; CR = 0.9;

    fprintf('[DE开始] 粗解计算中...\n');
    % 初始化种群
    X = zeros(pop,8);
    for i=1:pop
        X(i,:) = de_rand_vec();
    end

    fit = zeros(pop,1);
    for i=1:pop
        fit(i) = eval_one(X(i,:));
    end
    [best_fit, best_idx] = max(fit);
    best = X(best_idx,:);

    D = size(X,2);
    for g=1:gens
        for i=1:pop
            idxs = setdiff(1:pop, i);
            rp = randperm(numel(idxs),3);
            a = idxs(rp(1)); b = idxs(rp(2)); c = idxs(rp(3));
            Xa = X(a,:); Xb = X(b,:); Xc = X(c,:);
            mutant = Xa + F*(Xb - Xc) + 0.2*(best - Xa);
            cross  = zeros(1,D);
            jrand  = randi(D);
            for j=1:D
                if rand < CR || j==jrand
                    cross(j) = mutant(j);
                else
                    cross(j) = X(i,j);
                end
            end
            % 变量边界处理（与 Python 一致）
            cross(1) = mod(cross(1), 2*pi);
            cross(2) = min(max(cross(2),70),140);
            global SC
            for idx=[3,5,7], cross(idx)=min(max(cross(idx),0),SC.max_throw_time); end
            for idx=[4,6,8], cross(idx)=min(max(cross(idx),0),8); end

            val = eval_one(cross);
            if val >= fit(i)
                X(i,:) = cross; fit(i) = val;
                if val > best_fit
                    best_fit = val; best = cross;
                end
            end
        end
        fprintf('[DE] gen=%d/%d, best J=%.6f\n', g, gens, best_fit);
    end
    x_best = best; j_best = best_fit;
    fprintf('\n[DE结束] 粗解 J=%.6f\n', j_best);

    % ------------------ 局部抛光 ------------------
    fprintf('[局部抛光] 开始微调...\n');
    x_ref = x_best;
    steps = [deg2rad(6), 8.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8];
    rounds = 3;
    phi = (sqrt(5)-1)/2;

    for rr=1:rounds
        fprintf('[抛光] 回合 %d/%d\n', rr, rounds);
        for d=1:8
            a = -steps(d); b = steps(d);
            % 定义一维线搜索目标
            f = @(s) local_f(x_ref, d, s);
            c = b - phi*(b-a); d2 = a + phi*(b-a);
            [fc, xc] = f(c);
            [fd, xd] = f(d2);
            for iter=1:10
                if fc < fd
                    a = c; c = d2; fc = fd; xc = xd;
                    d2 = a + phi*(b-a);
                    [fd, xd] = f(d2);
                else
                    b = d2; d2 = c; fd = fc; xd = xc;
                    c = b - phi*(b-a);
                    [fc, xc] = f(c);
                end
            end
            [~, x_ref] = f(0.5*(a+b));
            steps(d) = steps(d)*0.5;
        end
    end

    % ------------------ 输出结果 ------------------
    th   = x_ref(1); v = x_ref(2);
    tr1  = x_ref(3); tau1 = x_ref(4);
    tr2  = x_ref(5); tau2 = x_ref(6);
    tr3  = x_ref(7); tau3 = x_ref(8);

    [J, pack] = objective_fn(th, v, [tr1,tr2,tr3], [tau1,tau2,tau3]);
    T_cov = pack.T_cov;
    bombs = pack.bombs;

    % 打印表头（中文）
    fprintf('\n===== 第3问 优化结果（DE+抛光） =====\n');
    header = [ ...
        '无人机方向(°)  无人机速度(m/s)  弹序号  ', ...
        '投放点x(m)     投放点y(m)     投放点z(m)     ', ...
        '起爆点x(m)     起爆点y(m)     起爆点z(m)     有效干扰时段(s)'];
    fprintf('%s\n', header);
    fprintf('%s\n', repmat('-',1,length(header)));

    for idx=1:3
        part = participate_intervals(bombs(idx), T_cov);
        part_str = format_intervals(part);
        r_launch = bombs(idx).r_launch; r_det = bombs(idx).r_det;
        fprintf('%12.6f  %16.6f  %6d  %12.6f  %12.6f  %12.6f  %12.6f  %12.6f  %12.6f  %s\n', ...
            rad2deg(th), v, idx, ...
            r_launch(1), r_launch(2), r_launch(3), ...
            r_det(1),    r_det(2),    r_det(3), ...
            part_str);
    end

    cov_str = format_intervals(T_cov);
    fprintf('\n整体严格遮蔽时间段：%s\n', cov_str);
    fprintf('总覆盖时间 J=%.6f s\n', union_len(T_cov));
end

% 一维线搜索辅助：在第 d 维做偏移 s
function [J, x2] = local_f(x_ref, d, s)
    global SC
    x2 = x_ref;
    x2(d) = x2(d) + s;
    % 变量边界处理
    x2(1) = mod(x2(1), 2*pi);
    x2(2) = min(max(x2(2),70),140);
    for idx=[3,5,7], x2(idx) = min(max(x2(idx),0), SC.max_throw_time); end
    for idx=[4,6,8], x2(idx) = min(max(x2(idx),0), 8); end
    [J, ~] = objective_fn(x2(1), x2(2), [x2(3),x2(5),x2(7)], [x2(4),x2(6),x2(8)]);
end

% 单个个体评价
function J = eval_one(x)
    th=x(1); v=x(2);
    tr1=x(3); tau1=x(4);
    tr2=x(5); tau2=x(6);
    tr3=x(7); tau3=x(8);
    [J,~] = objective_fn(th, v, [tr1,tr2,tr3], [tau1,tau2,tau3]);
end

% 随机向量（种群个体）
function x = de_rand_vec()
    global SC
    theta = 2*pi*rand();
    v     = 70 + (140-70)*rand();
    trels = sort(SC.max_throw_time*rand(1,3));
    % 投掷间隔约束
    if trels(2) < trels(1) + SC.min_throw_interval
        trels(2) = min(SC.max_throw_time, trels(1) + SC.min_throw_interval);
    end
    if trels(3) < trels(2) + SC.min_throw_interval
        trels(3) = min(SC.max_throw_time, trels(2) + SC.min_throw_interval);
    end
    taus  = 8*rand(1,3);
    x = [theta, v, trels(1), taus(1), trels(2), taus(2), trels(3), taus(3)];
end

% ===================== 目标函数 =====================
function [J, pack] = objective_fn(theta, v, trels, taus)
    % —— 约束投影 —— %
    [theta, v, trels, taus] = project_params(theta, v, trels, taus);

    % —— 用模板字段预分配结构体数组，确保字段一致 —— %
    bombs = repmat(struct( ...
        'r_launch', [], ...
        'r_det',    [], ...
        't_det',    [], ...
        'per_point',[], ...
        'own',      []), 1, 3);

    % —— 逐枚计算 —— %
    for k = 1:3
        b = bomb_intervals(theta, v, trels(k), taus(k));
        if isempty(b)
            J = -1e9; 
            pack = [];
            return;
        end
        bombs(k) = b;   % 现在字段一致，可正常赋值
    end

    % —— 严格并交与目标值 —— %
    [T_cov, Jlen] = strict_union_3(bombs(1), bombs(2), bombs(3));
    J = Jlen;

    % —— 打包输出 —— %
    pack.theta = theta; 
    pack.v     = v; 
    pack.trels = trels; 
    pack.taus  = taus;
    pack.bombs = bombs; 
    pack.T_cov = T_cov;
end

% =======================================================================
% 新增：同时优化对 M1、M2、M3 三枚导弹的总严格遮蔽时长
% =======================================================================

function run_q3_progress_multi()
    % 与单导弹版一致的 DE/抛光参数
    pop = 56; gens = 500; F = 0.7; CR = 0.9;

    fprintf('[DE开始][三导弹] 粗解计算中...\n');

    % 初始化种群
    X = zeros(pop,8);
    for i=1:pop
        X(i,:) = de_rand_vec();
    end

    fit = zeros(pop,1);
    for i=1:pop
        fit(i) = eval_one_multi(X(i,:));
    end
    [best_fit, best_idx] = max(fit);
    best = X(best_idx,:);

    D = size(X,2);
    for g=1:gens
        for i=1:pop
            idxs = setdiff(1:pop, i);
            rp = randperm(numel(idxs),3);
            a = idxs(rp(1)); b = idxs(rp(2)); c = idxs(rp(3));
            Xa = X(a,:); Xb = X(b,:); Xc = X(c,:);
            mutant = Xa + F*(Xb - Xc) + 0.2*(best - Xa);
            cross  = zeros(1,D);
            jrand  = randi(D);
            for j=1:D
                if rand < CR || j==jrand
                    cross(j) = mutant(j);
                else
                    cross(j) = X(i,j);
                end
            end
            % 变量边界处理
            global SC
            cross(1) = mod(cross(1), 2*pi);
            cross(2) = min(max(cross(2),70),140);
            for idx=[3,5,7], cross(idx)=min(max(cross(idx),0),SC.max_throw_time); end
            for idx=[4,6,8], cross(idx)=min(max(cross(idx),0),8); end

            val = eval_one_multi(cross);
            if val >= fit(i)
                X(i,:) = cross; fit(i) = val;
                if val > best_fit
                    best_fit = val; best = cross;
                end
            end
        end
        fprintf('[DE][三导弹] gen=%d/%d, best J_total=%.6f\n', g, gens, best_fit);
    end
    x_best = best; j_best = best_fit;
    fprintf('\n[DE结束][三导弹] 粗解 J_total=%.6f\n', j_best);

    % ------------------ 局部抛光 ------------------
    fprintf('[局部抛光][三导弹] 开始微调...\n');
    x_ref = x_best;
    steps = [deg2rad(6), 8.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8];
    rounds = 3;
    phi = (sqrt(5)-1)/2;

    for rr=1:rounds
        fprintf('[抛光][三导弹] 回合 %d/%d\n', rr, rounds);
        for d=1:8
            a = -steps(d); b = steps(d);
            f = @(s) local_f_multi(x_ref, d, s);
            c = b - phi*(b-a); d2 = a + phi*(b-a);
            [fc, xc] = f(c);
            [fd, xd] = f(d2);
            for iter=1:10
                if fc < fd
                    a = c; c = d2; fc = fd; xc = xd;
                    d2 = a + phi*(b-a);
                    [fd, xd] = f(d2);
                else
                    b = d2; d2 = c; fd = fc; xd = xc;
                    c = b - phi*(b-a);
                    [fc, xc] = f(c);
                end
            end
            [~, x_ref] = f(0.5*(a+b));
            steps(d) = steps(d)*0.5;
        end
    end

    th   = x_ref(1); v = x_ref(2);
    tr1  = x_ref(3); tau1 = x_ref(4);
    tr2  = x_ref(5); tau2 = x_ref(6);
    tr3  = x_ref(7); tau3 = x_ref(8);

    [J_total, pack] = objective_fn_multi(th, v, [tr1,tr2,tr3], [tau1,tau2,tau3]);

    % ------------------ 输出结果 ------------------
    fprintf('\n===== 第5问风格：FY1 对 M1/M2/M3 三导弹 总严格遮蔽时长最大化（DE+抛光） =====\n');
    fprintf('无人机方向(°)=%.6f, 无人机速度=%.6f m/s\n', rad2deg(th), v);

    % 分导弹输出
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

% ---------- 多导弹：个体评价 ----------
function J = eval_one_multi(x)
    th=x(1); v=x(2);
    tr1=x(3); tau1=x(4);
    tr2=x(5); tau2=x(6);
    tr3=x(7); tau3=x(8);
    [J,~] = objective_fn_multi(th, v, [tr1,tr2,tr3], [tau1,tau2,tau3]);
end

% ---------- 多导弹：一维线搜索步 ----------
function [J, x2] = local_f_multi(x_ref, d, s)
    global SC
    x2 = x_ref;
    x2(d) = x2(d) + s;
    % 变量边界处理
    x2(1) = mod(x2(1), 2*pi);
    x2(2) = min(max(x2(2),70),140);
    for idx=[3,5,7], x2(idx) = min(max(x2(idx),0), SC.max_throw_time); end
    for idx=[4,6,8], x2(idx) = min(max(x2(idx),0), 8); end
    [J, ~] = objective_fn_multi(x2(1), x2(2), [x2(3),x2(5),x2(7)], [x2(4),x2(6),x2(8)]);
end

% ---------- 多导弹：目标函数 ----------
function [J_total, pack] = objective_fn_multi(theta, v, trels, taus)
    % 与单导弹一样先做约束投影
    [theta, v, trels, taus] = project_params(theta, v, trels, taus);
    global SC
    pos_list = SC.missile_pos_list;

    bombs_all = cell(1,3);
    Tcov_all  = cell(1,3);
    Jsum = 0.0;

    for m = 1:3
        mpos0 = pos_list(m,:);                        % 本枚导弹初始位置
        % 针对本导弹计算三枚烟幕弹及严格遮蔽
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
        Jsum = Jsum + Jm;                              % 总和目标
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

% ===================== 约束投影 =====================
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

% ===================== 单点覆盖时间计算（边沿二分） =====================
function intervals = intervals_for_point(theta, v, t_rel, tau, idx)
    global SC
    [~, r_det, t_det] = detonation_point(theta, v, t_rel, tau);
    if ~(0 <= t_det && t_det <= SC.max_det_time)
        intervals = zeros(0,2); return;
    end
    ts = t_det:SC.dt:(t_det + SC.smoke_lifetime + 1e-12);
    if isempty(ts), intervals = zeros(0,2); return; end

    % 状态函数：该点是否被烟幕严格遮蔽
    function s = state(t)
        c = smoke_center(t, r_det, t_det);
        if isempty(c)
            s = false; return;
        end
        pts = two_sides();  % 采样点（圆柱上下表面+中心）
        s = segment_sphere_intersect(missile_position(t), pts(idx,:), c, SC.smoke_radius);
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
    % 删除空/退化区间并合并
    good = out(:,2) - out(:,1) > 1e-9;
    intervals = merge_intervals(out(good,:));
end

function b = bomb_intervals(theta, v, t_rel, tau)
    per_point = cell(1,6);
    for i=1:6
        per_point{i} = intervals_for_point(theta, v, t_rel, tau, i);
    end
    own_union = merge_intervals(vertcat(per_point{:}));
    [r_launch, r_det, t_det] = detonation_point(theta, v, t_rel, tau);
    b.r_launch = r_launch;
    b.r_det    = r_det;
    b.t_det    = t_det;
    b.per_point= per_point;
    b.own      = own_union;
end

function [T_cov, J] = strict_union_3(B1,B2,B3)
    unions = cell(1,6);
    for i=1:6
        Ui = merge_intervals([B1.per_point{i}; B2.per_point{i}; B3.per_point{i}]);
        unions{i} = Ui;
    end
    T_cov = intersect_many(unions);
    J = union_len(T_cov);
end

function part = participate_intervals(B, T_cov)
    part = intersect_two(T_cov, B.own);
end

% ===================== 基础几何与运动函数 =====================
function v = unit_v(x)
    n = sqrt(sum(x.^2));
    if n==0, v = x; else, v = x./n; end
end

function p = missile_position(t)
    global SC
    MISSILE_DIR = unit_v(SC.decoy_pos - SC.missile_pos0);
    p = SC.missile_pos0 + SC.missile_speed * MISSILE_DIR * t;
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
    if t < t_det
        c = []; return;
    end
    dt = t - t_det;
    if dt > SC.smoke_lifetime
        c = []; return;
    end
    c = r_d + [0.0, 0.0, -SC.smoke_v_down*dt];
end

function pts = two_sides()
    % 用圆柱上下表面均匀采样点替代六点法
    global SC
    num_radial = 8;
    num_height = 2;
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
    % 顶部中心 & 底部中心
    pts(idx,:)   = [TX, TY, Z0+H]; idx=idx+1;
    pts(idx,:)   = [TX, TY, Z0];
end

function tf = segment_sphere_intersect(p0, p1, c, R)
    d = p1 - p0; f = p0 - c;
    a = dot(d,d); b = 2*dot(f,d); c0 = dot(f,f) - R^2;
    disc = b*b - 4*a*c0;
    if disc < 0
        tf = false; return;
    end
    sdc = sqrt(max(0.0, disc));
    s1 = (-b - sdc)/(2*a);
    s2 = (-b + sdc)/(2*a);
    tf = (0.0 <= s1 && s1 <= 1.0) || (0.0 <= s2 && s2 <= 1.0);
end

% ===================== 时间区间操作 =====================
function intervals = merge_intervals(intervals)
    if isempty(intervals)
        intervals = zeros(0,2); return;
    end
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
            i = i+1;
        else
            j = j+1;
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

% ===================== 工具函数 =====================
function s = format_intervals(U)
    if isempty(U)
        s = '[]'; return;
    end
    parts = strings(size(U,1),1);
    for i=1:size(U,1)
        parts(i) = sprintf('[%.3f,%.3f]', U(i,1), U(i,2));
    end
    s = "[" + strjoin(parts', ', ') + "]";
    s = char(s);
end
