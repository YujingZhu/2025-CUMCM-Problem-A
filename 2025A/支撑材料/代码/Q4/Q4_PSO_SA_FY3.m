%% 问题2：单无人机单烟幕优化（FY1 对 M1，严格口径·仅采顶/底圆周）
% 目标：最大化对 M1 的“完全遮蔽”时间；R2024a
clear; clc; close all;

%% ====== 题面参数 ======
g=9.81; v_m=300; v_s=3; R_s=10; T_eff=20; R_t=7; H_t=10; dt=0.02;
O=[0,0,0]; C_t=[0,200,5]; M0=[20000,0,2000]; F0=[6000, -3000, 700];
u_m=-M0/norm(M0); T_max=norm(M0)/v_m;

%% ====== PSO 参数（已加长迭代 & 细致打印） ======
N_particles=300;              % 粒子数（可按机器性能调）
MaxIter=500;                 % ★延长：迭代次数
w=0.7; c1=1.5; c2=1.5;
lb=[0,70,0,0.10]; ub=[2*pi,140,30,8.00];

verbose=true;                % ★详尽打印开关（true：打印均值/方差/提升等）

fprintf('开始 PSO 优化: 粒子=%d, 迭代=%d\n',N_particles,MaxIter);

%% ====== 初始化 ======
rng(2024);
X=lb+(ub-lb).*rand(N_particles,4);
V=0.1*(ub-lb).*rand(N_particles,4);
pBest=X; pBestVal=inf(N_particles,1);
gBest=X(1,:); gBestVal=inf;
gBestHist=nan(MaxIter,1); meanFitHist=nan(MaxIter,1);

%% ====== 主循环（修正版） ======
for iter=1:MaxIter
    fitVals=zeros(N_particles,1);
    for p=1:N_particles
        fit=objectiveFunction(X(p,:),M0,F0,C_t,u_m,g,v_m,v_s,R_s,R_t,H_t,T_eff,dt);
        fitVals(p)=fit;
        if fit<pBestVal(p), pBest(p,:)=X(p,:); pBestVal(p)=fit; end
        if fit<gBestVal,   gBest=X(p,:); gBestVal=fit;         end
    end

    % 统计信息（适应度= -遮蔽时长）
    bestCover = -gBestVal;                  % 当前最优遮蔽时间（秒）
    gBestHist(iter)  = bestCover;
    meanFitHist(iter)= -mean(fitVals);
    medCover = median(-fitVals);
    stdCover = std(-fitVals);

    % ★ 这里是修正点：用 if-else 代替非法的 ?: 运算
    if iter==1
        dBest = NaN;                        % 第一轮没有“提升量”
    else
        dBest = bestCover - gBestHist(iter-1);
    end

    % 每轮详细打印
    fprintf(['Iter %3d | best=%.3f s | mean=%.3f | median=%.3f | std=%.3f | dBest=%+.4f ', ...
             '| theta=%.2f° v_u=%.2f t_rel=%.2f t_fuse=%.2f\n'], ...
             iter,bestCover,meanFitHist(iter),medCover,stdCover,dBest, ...
             gBest(1)*180/pi,gBest(2),gBest(3),gBest(4));

    % 速度与位置更新
    for p=1:N_particles
        r1=rand(1,4); r2=rand(1,4);
        V(p,:)=w*V(p,:)+c1*r1.*(pBest(p,:)-X(p,:))+c2*r2.*(gBest-X(p,:));
        Vmax=0.2*(ub-lb); V(p,:)=max(min(V(p,:),Vmax),-Vmax);
        X(p,:)=X(p,:)+V(p,:); X(p,:)=max(min(X(p,:),ub),lb);
    end
end


%% ====== 输出最优结果 ======
fprintf('\n=== 优化完成（严格口径·圆周采样）===\n');
fprintf('  theta1 = %.4f rad (%.2f°)\n', gBest(1), gBest(1)*180/pi);
fprintf('  v_u    = %.2f m/s\n',        gBest(2));
fprintf('  t_rel  = %.3f s\n',          gBest(3));
fprintf('  t_fuse = %.3f s\n',          gBest(4));

[fitness_star,P_drop,C_e,t_e]=objectiveFunction(gBest,M0,F0,C_t,u_m,g,v_m,v_s,R_s,R_t,H_t,T_eff,dt);
fprintf('\n关键位置：\n');
fprintf('  P_drop = (%.2f, %.2f, %.2f) m\n',P_drop);
fprintf('  C_e    = (%.2f, %.2f, %.2f) m\n',C_e);
fprintf('  t_e    = %.3f s\n',t_e);
fprintf('\n最大“完全遮蔽”时长：%.3f s\n',-fitness_star);

%% ====== 图1：收敛曲线（每轮最优） ======
figure('Position',[200 200 560 420]);
plot(1:MaxIter,gBestHist,'LineWidth',1.8); grid on;
xlabel('粒子群迭代次数'); ylabel('全局最优遮蔽时间 / s'); title('粒子群算法收敛过程'); xlim([1 MaxIter]);

%% ====== 图2：三维世界全景（无人机/烟雾弹/云团/导弹/真目标/假目标） ======
visualizeResult(gBest,M0,F0,C_t,u_m,g,v_m,v_s,R_s,R_t,H_t,T_eff,dt,O,T_max);

%% =================== 目标函数 ===================
function [fitness,P_drop,C_e,t_e]=objectiveFunction(x,M0,F0,C_t,u_m,g,v_m,v_s,R_s,R_t,H_t,T_eff,dt)
    theta1=x(1); v_u=x(2); t_rel=x(3); t_fuse=x(4);
    dir=[cos(theta1),sin(theta1),0];
    P_drop=F0+v_u*t_rel*dir; v_drop=v_u*dir;
    C_e=P_drop+v_drop*t_fuse; C_e(3)=C_e(3)-0.5*g*t_fuse^2; t_e=t_rel+t_fuse;
    if C_e(3)<=0 || t_e<0, fitness=1e3; return; end
    T_cover=0; t_vec=t_e:dt:(t_e+T_eff);
    for t=t_vec
        M_pos=M0+v_m*t*u_m; C_cloud=C_e-[0,0,v_s*(t-t_e)];
        if isOccluded_rims_strict(M_pos,C_t,C_cloud,R_t,H_t,R_s)
            T_cover=T_cover+dt;
        end
    end
    fitness=-T_cover;
end

%% 遮蔽判定（仅顶/底圆周）
function occluded=isOccluded_rims_strict(M_pos,C_t_center,C_cloud,R_t,H_t,R_s)
    C_base=[C_t_center(1),C_t_center(2),C_t_center(3)-H_t/2];
    nTheta=360;
    rimPts=sampleCylinderRims(C_base,R_t,H_t,nTheta);
    occluded=true;
    for k=1:size(rimPts,1)
        P=rimPts(k,:);
        if ~segmentSphereIntersect(M_pos,P,C_cloud,R_s)
            occluded=false; return;
        end
    end
end

%% 圆柱顶/底圆周采样
function rimPts=sampleCylinderRims(C_base,R,H,nTheta)
    cx=C_base(1); cy=C_base(2); cz=C_base(3);
    z_bot=cz; z_top=cz+H;
    th=linspace(0,2*pi,nTheta+1); th(end)=[];
    rimPts=zeros(2*nTheta,3);
    for i=1:nTheta
        c=cos(th(i)); s=sin(th(i));
        rimPts(2*i-1,:)=[cx+R*c, cy+R*s, z_bot];
        rimPts(2*i  ,:)=[cx+R*c, cy+R*s, z_top];
    end
end

%% 线段-球相交（二次方程）
function hit=segmentSphereIntersect(M,P,C,r)
    d=P-M; f=M-C; A=dot(d,d);
    if A<=1e-16, hit=(dot(f,f)<=r*r+1e-12); return; end
    B=2*dot(f,d); Q=dot(f,f)-r*r; D=B*B-4*A*Q;
    if D<=0, hit=false; return; end
    sqrtD=sqrt(D); t1=(-B - sqrtD)/(2*A); t2=(-B + sqrtD)/(2*A);
    t_enter=min(t1,t2); t_exit=max(t1,t2);
    hit=(t_exit>=0.0)&&(t_enter<=1.0);
end

%% 三维可视化（无人机/烟雾弹/云团/导弹/真目标/假目标）
function visualizeResult(gBest,M0,F0,C_t,u_m,g,v_m,v_s,R_s,R_t,H_t,T_eff,dt,O,T_max)
    figure('Position',[100,100,1200,850]); hold on; grid on; axis equal; view(45,28);
    [~,P_drop,C_e,t_e]=objectiveFunction(gBest,M0,F0,C_t,u_m,g,v_m,v_s,R_s,R_t,H_t,T_eff,dt);
    theta1=gBest(1); v_u=gBest(2); t_rel=gBest(3); t_fuse=gBest(4); dir=[cos(theta1),sin(theta1),0];
    t_missile=linspace(0,T_max,400); M_traj=M0+(v_m.*t_missile').*u_m;
    plot3(M_traj(:,1),M_traj(:,2),M_traj(:,3),'r-','LineWidth',2,'DisplayName','导弹轨迹');
    t_drone=linspace(0,t_rel,200); F_traj=F0+(v_u.*t_drone').*dir;
    plot3(F_traj(:,1),F_traj(:,2),F_traj(:,3),'b-','LineWidth',2,'DisplayName','无人机轨迹');
    t_ball=linspace(0,t_fuse,150); v_drop=v_u*dir;
    Ball_traj=P_drop+(v_drop.*t_ball')+[zeros(numel(t_ball),2),-0.5*g*(t_ball').^2];
    plot3(Ball_traj(:,1),Ball_traj(:,2),Ball_traj(:,3),'m-','LineWidth',2,'DisplayName','烟雾弹抛射轨迹');
    [xs,ys,zs]=sphere(44);
    surf(C_e(1)+R_s*xs,C_e(2)+R_s*ys,C_e(3)+R_s*zs,'FaceAlpha',0.25,'EdgeColor','none','DisplayName','烟幕球（起爆瞬时）');
    t_cloud=linspace(0,T_eff,80); C_cloud=C_e-[zeros(numel(t_cloud),2), v_s*t_cloud'];
    plot3(C_cloud(:,1),C_cloud(:,2),C_cloud(:,3),'c--','LineWidth',1.5,'DisplayName','云团中心下沉');
    plot3(C_cloud(end,1),C_cloud(end,2),C_cloud(end,3),'co','MarkerFaceColor','c','DisplayName','云团有效期末中心');
    C_base=[C_t(1),C_t(2),C_t(3)-H_t/2]; drawCylinderSolid(C_base,R_t,H_t,[0.3 0.7 0.3],0.35,'真目标（圆柱）');
    plot3(O(1),O(2),O(3),'kp','MarkerFaceColor','y','MarkerSize',10,'DisplayName','假目标（原点）');
    L=20; quiver3(0,0,0, L,0,0,'k','LineWidth',1,'MaxHeadSize',2,'DisplayName','x轴');
    quiver3(0,0,0, 0,L,0,'k','LineWidth',1,'MaxHeadSize',2,'DisplayName','y轴');
    quiver3(0,0,0, 0,0,L,'k','LineWidth',1,'MaxHeadSize',2,'DisplayName','z轴');
    plot3(M0(1),M0(2),M0(3),'ro','MarkerFaceColor','r','DisplayName','M1 起点');
    plot3(F0(1),F0(2),F0(3),'bo','MarkerFaceColor','b','DisplayName','FY1 起点');
    plot3(P_drop(1),P_drop(2),P_drop(3),'ms','MarkerFaceColor','m','DisplayName','投放点');
    plot3(C_e(1),C_e(2),C_e(3),'co','MarkerFaceColor','c','DisplayName','起爆点');
    plot3(C_t(1),C_t(2),C_t(3),'g^','MarkerFaceColor','g','DisplayName','真目标中心');
    xlabel('X / m'); ylabel('Y / m'); zlabel('Z / m'); title('FY1 对 M1 · 烟幕干扰最优策略（严格口径·仅顶/底圆周）');
    legend('Location','northeastoutside'); axis vis3d;
end

%% 画实心圆柱
function drawCylinderSolid(C_base,R,H,colorRGB,alphaVal,dispName)
    [xc,yc,zc]=cylinder(R,80); zc=zc*H+C_base(3); xc=xc+C_base(1); yc=yc+C_base(2);
    s1=surf(xc,yc,zc,'FaceColor',colorRGB,'FaceAlpha',alphaVal,'EdgeColor','none','DisplayName',dispName);
    th=linspace(0,2*pi,120);
    xTop=C_base(1)+R*cos(th); yTop=C_base(2)+R*sin(th); zTop=C_base(3)+H*ones(size(th));
    xBot=C_base(1)+R*cos(th); yBot=C_base(2)+R*sin(th); zBot=C_base(3)*ones(size(th));
    p1=patch(xTop,yTop,zTop,colorRGB,'FaceAlpha',alphaVal,'EdgeColor','none','HandleVisibility','off');
    p2=patch(xBot,yBot,zBot,colorRGB,'FaceAlpha',alphaVal,'EdgeColor','none','HandleVisibility','off');
    uistack([s1 p1 p2],'bottom');
end
