% 三无人机协同优化（扫描+二分法）
clear; clc; close all;

%% 参数设置
env.g = 9.81; env.v_dandao = 300; env.v_yanwu = 3; env.R_yanwu = 10; env.T_eff = 20;
env.R_t = 7; env.H_t = 10; env.dt = 0.05;
env.C_t = [0,200,5]; env.M0 = [20000,0,2000]; env.u_dandao = -env.M0/norm(env.M0);

drones = {[17800,0,1800],[12000,1400,1400],[6000,-3000,700]};

fprintf('开始扫描+二分法快速优化...\n'); tic;

%% 粗扫描+二分法
results_all = cell(3,1);

for iD = 1:3
    F0 = drones{iD};
    
    thetaR = linspace(0,2*pi,36);
    vR = linspace(80,130,11);
    tR = linspace(2,25,12);
    fuseR = linspace(1,6,11);
    
    bestVal = -inf; bestP = [];
    
    totalLoop = numel(thetaR)*numel(vR)*numel(tR)*numel(fuseR);
    cnt = 0;
    for th=thetaR
        for v=vR
            for t_rel=tR
                for fuse=fuseR
                    cnt = cnt + 1;
                    if mod(cnt,1000)==0
                        fprintf('  FY%d 进度: %.1f%%\n',iD,100*cnt/totalLoop);
                    end
                    val = fastEvaluate([th,v,t_rel,fuse],F0,env);
                    if val>bestVal
                        bestVal = val;
                        bestP = [th,v,t_rel,fuse];
                    end
                end
            end
        end
    end
    fprintf('  FY%d 粗扫描完成: %.3f s\n', iD, bestVal);
    
    refined = refineBinary(bestP,F0,env);
    finalScore = highPrecisionEvaluate(refined,F0,env);
    
    fprintf('  FY%d 精细优化完成: %.3f s\n', iD, finalScore);
    
    results_all{iD} = struct('params',refined,'score',finalScore,'F0',F0);
end

%% 输出结果
total_score = 0;
droneNames = {'FY1','FY2','FY3'};
resTable = cell(3,11);

for iD = 1:3
    p = results_all{iD}.params; F0 = results_all{iD}.F0;
    dir = [cos(p(1)),sin(p(1)),0];
    P_drop = F0 + p(2)*p(3)*dir;
    C_explode = P_drop + p(2)*dir*p(4); C_explode(3) = C_explode(3)-0.5*env.g*p(4)^2;
    
    resTable(iD,:) = {droneNames{iD},p(2),mod(rad2deg(p(1)),360),p(3),p(4), ...
                      P_drop(1),P_drop(2),P_drop(3),C_explode(1),C_explode(2),C_explode(3)};
    total_score = total_score + results_all{iD}.score;
end

fprintf('\n总遮蔽时长: %.3f s\n', total_score);

% 核心函数

function val = fastEvaluate(p,F0,env)
    th = p(1); v = p(2); t_rel = p(3); fuse = p(4);
    if v<70 || v>140 || t_rel<0 || t_rel>30 || fuse<0.1 || fuse>8
        val = 0; return;
    end
    dir = [cos(th),sin(th),0];
    P_drop = F0 + v*t_rel*dir;
    C_e = P_drop + v*dir*fuse; C_e(3) = C_e(3)-0.5*env.g*fuse^2;
    t_e = t_rel + fuse;
    if C_e(3)<=0 || t_e<0; val=0; return; end
    
    val=0;
    for t=t_e:env.dt:(t_e+env.T_eff)
        Mpos = env.M0 + env.v_dandao*t*env.u_dandao;
        Ccloud = C_e - [0,0,env.v_yanwu*(t-t_e)];
        if occludedFast(Mpos,env.C_t,Ccloud,env.R_t,env.H_t,env.R_yanwu)
            val = val + env.dt;
        end
    end
end

function flag = occludedFast(Mpos,Ct,Cc,Rt,Ht,Rs)
    Cb = [Ct(1),Ct(2),Ct(3)-Ht/2]; nTheta=12;
    pts = sampleRim(Cb,Rt,Ht,nTheta);
    flag = true;
    for k=1:size(pts,1)
        if ~segSphere(Mpos,pts(k,:),Cc,Rs)
            flag=false; return;
        end
    end
end

function pts = sampleRim(Cb,R,H,n)
    thetas=linspace(0,2*pi,n+1); thetas(end)=[];
    pts=zeros(2*n,3);
    for i=1:n
        c=cos(thetas(i)); s=sin(thetas(i));
        pts(2*i-1,:)=[Cb(1)+R*c,Cb(2)+R*s,Cb(3)];
        pts(2*i,:)  =[Cb(1)+R*c,Cb(2)+R*s,Cb(3)+H];
    end
end

function hit = segSphere(M,P,C,r)
    d = P-M; f=M-C; A=dot(d,d);
    if A<=1e-16; hit=(dot(f,f)<=r^2+1e-12); return; end
    B=2*dot(f,d); Q=dot(f,f)-r^2;
    D=B*B-4*A*Q;
    if D<=0; hit=false; return; end
    t1=(-B-sqrt(D))/(2*A); t2=(-B+sqrt(D))/(2*A);
    hit = (min(t1,t2)<=1) && (max(t1,t2)>=0);
end

function refined = refineBinary(p,F0,env)
    refined = p; step=[pi/36,2.5,0.5,0.25];
    for r=1:3
        for idx=1:4
            s=step(idx)/r; phi=(1+sqrt(5))/2;
            cur = refined;
            for iter=1:12
                L=cur(idx)-s; R=cur(idx)+s;
                m1=R-(R-L)/phi; m2=L+(R-L)/phi;
                p1=cur; p2=cur; p1(idx)=m1; p2(idx)=m2;
                v1=highPrecisionEvaluate(p1,F0,env);
                v2=highPrecisionEvaluate(p2,F0,env);
                if v1>v2, cur(idx)=m1; else, cur(idx)=m2; end
                s=(R-L)/2; if s<1e-6, break; end
            end
            if highPrecisionEvaluate(cur,F0,env)>highPrecisionEvaluate(refined,F0,env)
                refined=cur;
            end
        end
    end
    refined = jointOpt(refined,F0,env);
end

function val = highPrecisionEvaluate(p,F0,env)
    dt=0.01; th=p(1); v=p(2); t_rel=p(3); fuse=p(4);
    if v<70||v>140||t_rel<0||t_rel>30||fuse<0.1||fuse>8, val=0; return; end
    dir=[cos(th),sin(th),0]; P_drop=F0+v*t_rel*dir; C_e=P_drop+v*dir*fuse; C_e(3)=C_e(3)-0.5*env.g*fuse^2;
    t_e=t_rel+fuse; if C_e(3)<=0||t_e<0, val=0; return; end
    val=0;
    for t=t_e:dt:(t_e+env.T_eff)
        Mpos=env.M0+env.v_dandao*t*env.u_dandao; Ccloud=C_e-[0,0,env.v_yanwu*(t-t_e)];
        if occludedHigh(Mpos,env.C_t,Ccloud,env.R_t,env.H_t,env.R_yanwu)
            val=val+dt;
        end
    end
end

function flag = occludedHigh(Mpos,Ct,Cc,Rt,Ht,Rs)
    Cb=[Ct(1),Ct(2),Ct(3)-Ht/2]; n=72;
    pts=sampleRim(Cb,Rt,Ht,n); flag=true;
    for k=1:size(pts,1)
        if ~segSphere(Mpos,pts(k,:),Cc,Rs), flag=false; return; end
    end
end

function best = jointOpt(p,F0,env)
    best=p; step=[pi/180,0.1,0.01,0.01];
    bestVal=highPrecisionEvaluate(p,F0,env);
    for it=1:50
        improve=false;
        for idx=1:4
            for dir=[-1,1]
                tmp=best; tmp(idx)=tmp(idx)+dir*step(idx);
                if idx==1, tmp(1)=mod(tmp(1),2*pi);
                elseif idx==2, if tmp(2)<70||tmp(2)>140, continue; end
                elseif idx==3, if tmp(3)<0||tmp(3)>30, continue; end
                elseif idx==4, if tmp(4)<0.1||tmp(4)>8, continue; end
                end
                v=highPrecisionEvaluate(tmp,F0,env);
                if v>bestVal, best=tmp; bestVal=v; improve=true; end
            end
        end
        if ~improve, step=step*0.8; if max(step)<1e-6, break; end; end
    end
end
