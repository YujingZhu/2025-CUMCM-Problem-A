# -*- coding: utf-8 -*-
"""
Q3：FY1 投放三枚干扰弹拦截 M1 导弹

"""

import math, random
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

# ===================== 场景与参数定义 =====================
@dataclass
class Scenario:
    # 目标（竖直圆柱体）
    target_x: float = 0.0
    target_y: float = 200.0
    target_z0: float = 0.0
    cyl_radius: float = 7.0
    cyl_height: float = 10.0

    # 烟幕参数
    smoke_radius: float = 10.0      # 烟幕球半径
    smoke_v_down: float = 3.0       # 烟幕下沉速度
    smoke_lifetime: float = 20.0    # 烟幕有效时间

    # 重力加速度
    g: float = 9.8

    # 无人机、导弹、假目标初始状态
    decoy_pos: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=float)
    missile_pos0: np.ndarray = np.array([20000.0, 0.0, 2000.0], dtype=float)
    missile_speed: float = 300.0
    fy1_pos0: np.ndarray = np.array([17800.0, 0.0, 1800.0], dtype=float)

    # 时间设置
    dt: float = 0.01
    refine_times: int = 10          # 边沿二分精度
    max_det_time: float = 35.0      # 起爆时间上限
    max_throw_time =67 
    min_throw_interval: float = 1.0 # 投掷间隔约束

SC = Scenario()

# ===================== 基础几何与运动函数 =====================
def norm(v): return float(np.linalg.norm(v))
def unit(v): n=norm(v); return v if n==0 else v/n

MISSILE_DIR = unit(SC.decoy_pos - SC.missile_pos0)

def missile_position(t):
    """导弹位置随时间变化"""
    return SC.missile_pos0 + SC.missile_speed * MISSILE_DIR * t

def fy1_position(t, theta, v):
    """无人机位置随时间变化"""
    dir_xy = np.array([math.cos(theta), math.sin(theta), 0.0])
    return SC.fy1_pos0 + v * dir_xy * t

def detonation_point(theta, v, t_rel, tau):
    """计算烟幕弹投放点、起爆点及起爆时间"""
    r_launch = fy1_position(t_rel, theta, v)
    accel = np.array([0.0, 0.0, -SC.g])
    r_detonation = r_launch + np.array([v*math.cos(theta), v*math.sin(theta), 0.0])*tau + 0.5*accel*(tau**2)
    t_det = t_rel + tau
    return r_launch, r_detonation, t_det

def smoke_center(t, r_d, t_det):
    """返回烟幕球心位置，如果已消失返回 None"""
    if t < t_det: return None
    dt = t - t_det
    if dt > SC.smoke_lifetime: return None
    return r_d + np.array([0.0, 0.0, -SC.smoke_v_down*dt])

# ===================== 严格遮蔽判定 =====================
def two_sides(missile):
    """
        用圆柱上下表面均匀采样点替代六点法
        missile: 导弹当前位置 (np.array)
        num_radial: 每层圆周采样点数
        num_height: 高度采样层数 (默认上下两层)
        返回采样点列表

        """
    
    num_radial = 8   # 每层圆周采样点数
    num_height = 2   # 高度层数（上下表面）
    TX, TY, Z0, R, H = SC.target_x, SC.target_y, SC.target_z0, SC.cyl_radius, SC.cyl_height
    points = []

    # 高度采样
    zs = np.linspace(Z0, Z0 + H, num_height)
    
    for z in zs:
        for i in range(num_radial):
            theta = 2 * math.pi * i / num_radial
            x = TX + R * math.cos(theta)
            y = TY + R * math.sin(theta)
            points.append(np.array([x, y, z], dtype=float))
    
    # 顶部中心点
    points.append(np.array([TX, TY, Z0 + H], dtype=float))
    # 底部中心点
    points.append(np.array([TX, TY, Z0], dtype=float))

    return points

def segment_sphere_intersect(p0,p1,c,R):
    """线段与球体相交判断"""
    d = p1-p0; f=p0-c
    a=float(np.dot(d,d)); b=2*float(np.dot(f,d)); c0=float(np.dot(f,f))-R**2
    disc = b*b - 4*a*c0
    if disc < 0.0: return False
    sdc = math.sqrt(max(0.0, disc))
    s1 = (-b - sdc)/(2*a); s2 = (-b + sdc)/(2*a)
    return (0.0 <= s1 <= 1.0) or (0.0 <= s2 <= 1.0)

# ===================== 时间区间操作 =====================
def merge_intervals(intervals, eps=1e-9):
    if not intervals: return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for a,b in intervals[1:]:
        la, lb = merged[-1]
        if a <= lb+eps: merged[-1][1] = max(lb,b)
        else: merged.append([a,b])
    return [(a,b) for a,b in merged]

def intersect_two(A,B):
    i=j=0; out=[]
    while i<len(A) and j<len(B):
        a1,b1=A[i]; a2,b2=B[j]
        a=max(a1,a2); b=min(b1,b2)
        if a<b: out.append((a,b))
        if b1<b2: i+=1
        else: j+=1
    return out

def intersect_many(list_of_intervals):
    cur = list_of_intervals[0]
    for U in list_of_intervals[1:]:
        cur = intersect_two(cur,U)
        if not cur: break
    return cur

def union_len(U): return sum(b-a for a,b in U)

# ===================== 单点覆盖时间计算（边沿二分） =====================
def intervals_for_point(theta,v,t_rel,tau,idx):
    r_launch, r_det, t_det = detonation_point(theta,v,t_rel,tau)
    if not (0 <= t_det <= SC.max_det_time): return []
    ts = np.arange(t_det, t_det+SC.smoke_lifetime+1e-12, SC.dt)
    def state(t):
        c = smoke_center(t, r_det, t_det)
        if c is None: return False
        pts = two_sides(missile_position(t))
        return segment_sphere_intersect(missile_position(t), pts[idx], c, SC.smoke_radius)
    out=[]; prev_s=state(ts[0])
    if prev_s: out.append([ts[0], None])
    for k in range(1,len(ts)):
        t = ts[k]; s = state(t)
        if s != prev_s:
            lo, hi, slo = ts[k-1], t, prev_s
            for _ in range(SC.refine_times):
                mid = 0.5*(lo+hi)
                sm = state(mid)
                if sm==slo: lo=mid
                else: hi=mid
            tsw = 0.5*(lo+hi)
            if not prev_s and s: out.append([tsw,None])
            elif prev_s and not s: 
                if out and out[-1][1] is None: out[-1][1]=tsw
        prev_s = s
    if prev_s and out and out[-1][1] is None: out[-1][1]=ts[-1]
    return merge_intervals([(a,b) for a,b in out if b>a+1e-9])

def bomb_intervals(theta,v,t_rel,tau):
    per_point = [intervals_for_point(theta,v,t_rel,tau,i) for i in range(6)]
    own_union = merge_intervals([iv for L in per_point for iv in L])
    r_launch,r_det,t_det = detonation_point(theta,v,t_rel,tau)
    return {"r_launch":r_launch, "r_det":r_det, "t_det":t_det, "per_point":per_point, "own":own_union}

def strict_union_3(B1,B2,B3):
    unions=[]
    for i in range(6):
        Ui = merge_intervals(B1["per_point"][i] + B2["per_point"][i] + B3["per_point"][i])
        unions.append(Ui)
    T_cov = intersect_many(unions)
    return T_cov, union_len(T_cov)

def participate_intervals(B,T_cov):
    return intersect_two(T_cov,B["own"])

# ===================== 约束投影 =====================
def project(theta,v,trels,taus):
    theta = theta % (2*math.pi)
    v = min(max(v,70.0),140.0)
    trels = [min(max(0.0,t), SC.max_throw_time) for t in trels]
    trels = sorted(trels)
    for i in range(1,3):
        if trels[i]<trels[i-1]+SC.min_throw_interval:
            trels[i]=min(SC.max_throw_time, trels[i-1]+SC.min_throw_interval)
    taus = [max(0.0,x) for x in taus]
    for k in range(3):
        t_det = trels[k]+taus[k]
        if t_det>SC.max_det_time:
            over = t_det - SC.max_det_time
            dec_tau = min(taus[k],over); taus[k]-=dec_tau; over-=dec_tau
            if over>1e-9: trels[k]=max(0.0, trels[k]-over)
    return theta,v,trels,taus

# ===================== 目标函数 =====================
def objective(theta,v,trels,taus):
    theta,v,trels,taus = project(theta,v,trels,taus)
    bombs=[]
    for k in range(3):
        b = bomb_intervals(theta,v,trels[k],taus[k])
        if b is None: return -1e9,None
        bombs.append(b)
    T_cov,J = strict_union_3(bombs[0],bombs[1],bombs[2])
    return J, (theta,v,trels,taus,bombs,T_cov)

# ===================== 差分进化算法 =====================
class DE:
    def __init__(self,pop=48,gens=240,F=0.7,CR=0.85,seed=2025):
        self.pop,self.gens,self.F,self.CR = pop,gens,F,CR
        random.seed(seed); np.random.seed(seed)

    def rand_vec(self):
        theta=random.uniform(0,2*math.pi)
        v=random.uniform(70,140)
        trels=sorted([random.uniform(0,SC.max_throw_time) for _ in range(3)])
        for i in [1,2]:
            if trels[i]<trels[i-1]+SC.min_throw_interval:
                trels[i]=min(SC.max_throw_time, trels[i-1]+SC.min_throw_interval)
        taus=[random.uniform(0,8) for _ in range(3)]
        return np.array([theta,v,trels[0],taus[0],trels[1],taus[1],trels[2],taus[2]])

    def run(self):
        X = np.vstack([self.rand_vec() for _ in range(self.pop)])
        fit = np.zeros(self.pop)
        def eval_one(x):
            th,v,tr1,tau1,tr2,tau2,tr3,tau3 = x.tolist()
            J,_ = objective(th,v,[tr1,tr2,tr3],[tau1,tau2,tau3])
            return J
        for i in range(self.pop): fit[i]=eval_one(X[i])
        best_idx=int(np.argmax(fit)); best=X[best_idx].copy(); best_fit=fit[best_idx]

        D=X.shape[1]
        for g in range(self.gens):
            for i in range(self.pop):
                idxs=list(range(self.pop)); idxs.remove(i)
                a,b,c=random.sample(idxs,3)
                Xa,Xb,Xc=X[a],X[b],X[c]
                mutant=Xa+self.F*(Xb-Xc)+0.2*(best-Xa)
                cross=np.empty(D); jrand=random.randrange(D)
                for j in range(D):
                    cross[j]=mutant[j] if random.random()<self.CR or j==jrand else X[i][j]
                cross[0]=cross[0]%(2*math.pi)
                cross[1]=min(max(cross[1],70),140)
                for idx in [2,4,6]: cross[idx]=min(max(cross[idx],0),SC.max_throw_time)
                for idx in [3,5,7]: cross[idx]=min(max(cross[idx],0),8)
                val=eval_one(cross)
                if val>=fit[i]:
                    X[i]=cross; fit[i]=val
                    if val>best_fit: best_fit=val; best=cross.copy()
            print(f"[DE] gen={g+1}/{self.gens}, best J={best_fit:.4f}")
        return best,best_fit

# ===================== 局部抛光 =====================
def polish(x,rounds=3):
    def pack(x):
        th,v,tr1,tau1,tr2,tau2,tr3,tau3 = x.tolist()
        return objective(th,v,[tr1,tr2,tr3],[tau1,tau2,tau3])
    steps=[math.radians(6),8.0,0.8,0.8,0.8,0.8,0.8,0.8]
    for _ in range(rounds):
        for d in range(8):
            a,b=-steps[d],steps[d]; phi=(math.sqrt(5)-1)/2
            def f(s):
                x2=x.copy(); x2[d]+=s
                x2[0]=x2[0]%(2*math.pi); x2[1]=min(max(x2[1],70),140)
                for i in [2,4,6]: x2[i]=min(max(x2[i],0),SC.max_throw_time)
                for i in [3,5,7]: x2[i]=min(max(x2[i],0),8)
                J,_=pack(x2); return J,x2
            c,d2=b-phi*(b-a),a+phi*(b-a); fc,xc=f(c); fd,xd=f(d2)
            for _ in range(10):
                if fc<fd: a,c,fc,xc=c,d2,fd,xd; d2=a+phi*(b-a); fd,xd=f(d2)
                else: b,d2,fd,xd=d2,c,fc,xc; c=b-phi*(b-a); fc,xc=f(c)
            _,x=f(0.5*(a+b)); steps[d]*=0.5
    return x

# ===================== 主程序 =====================
def run_q3_progress():
    from tqdm import trange, tqdm

    # ------------------ DE 粗解 ------------------
    de = DE(pop=56, gens=500, F=0.7, CR=0.9)

    print("[DE开始] 粗解计算中...")
    # 修改 DE.run() 以接收进度条
    X = np.vstack([de.rand_vec() for _ in range(de.pop)])
    fit = np.zeros(de.pop)
    def eval_one(x):
        th, v, tr1, tau1, tr2, tau2, tr3, tau3 = x.tolist()
        J,_ = objective(th,v,[tr1,tr2,tr3],[tau1,tau2,tau3])
        return J

    for i in range(de.pop): fit[i] = eval_one(X[i])
    best_idx = int(np.argmax(fit))
    best = X[best_idx].copy()
    best_fit = fit[best_idx]
    D = X.shape[1]

    for g in trange(de.gens, desc="[DE] 迭代进度"):
        for i in range(de.pop):
            idxs = list(range(de.pop))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)
            Xa, Xb, Xc = X[a], X[b], X[c]
            mutant = Xa + de.F*(Xb-Xc) + 0.2*(best-Xa)
            cross = np.empty(D)
            jrand = random.randrange(D)
            for j in range(D):
                cross[j] = mutant[j] if random.random()<de.CR or j==jrand else X[i][j]
            cross[0] = cross[0]%(2*math.pi)
            cross[1] = min(max(cross[1],70),140)
            for idx in [2,4,6]: cross[idx] = min(max(cross[idx],0),SC.max_throw_time)
            for idx in [3,5,7]: cross[idx] = min(max(cross[idx],0),8)
            val = eval_one(cross)
            if val >= fit[i]:
                X[i] = cross; fit[i] = val
                if val > best_fit:
                    best_fit = val; best = cross.copy()
    x_best, j_best = best, best_fit
    print(f"\n[DE结束] 粗解 J={j_best:.4f}")

    # ------------------ 局部抛光 ------------------
    print("[局部抛光] 开始微调...")
    x_ref = x_best.copy()
    steps = [math.radians(6),8.0,0.8,0.8,0.8,0.8,0.8,0.8]
    rounds = 3
    for _ in trange(rounds, desc="[抛光] 回合"):
        for d in range(8):
            a, b = -steps[d], steps[d]; phi = (math.sqrt(5)-1)/2
            def f(s):
                x2 = x_ref.copy()
                x2[d] += s
                x2[0] = x2[0]%(2*math.pi); x2[1]=min(max(x2[1],70),140)
                for i in [2,4,6]: x2[i] = min(max(x2[i],0),SC.max_throw_time)
                for i in [3,5,7]: x2[i] = min(max(x2[i],0),8)
                J,_=objective(x2[0],x2[1],[x2[2],x2[4],x2[6]],[x2[3],x2[5],x2[7]])
                return J, x2
            c,d2=b-phi*(b-a),a+phi*(b-a); fc,xc=f(c); fd,xd=f(d2)
            for _ in range(10):
                if fc<fd: a,c,fc,xc=c,d2,fd,xd; d2=a+phi*(b-a); fd,xd=f(d2)
                else: b,d2,fd,xd=d2,c,fc,xc; c=b-phi*(b-a); fc,xc=f(c)
            _,x_ref=f(0.5*(a+b))
            steps[d]*=0.5

    # ------------------ 输出结果 ------------------
    th,v,tr1,tau1,tr2,tau2,tr3,tau3 = x_ref.tolist()
    J,pack = objective(th,v,[tr1,tr2,tr3],[tau1,tau2,tau3])
    th,v,trels,taus,bombs,T_cov=pack

    import pandas as pd
    rows=[]
    for idx in range(3):
        part = participate_intervals(bombs[idx],T_cov)
        part_str = "[" + ", ".join([f"[{a:.3f},{b:.3f}]" for a,b in part]) + "]"
        r_launch,r_det=bombs[idx]["r_launch"],bombs[idx]["r_det"]
        rows.append({
            "无人机方向(°)": round(math.degrees(th),6),
            "无人机速度(m/s)": round(v,6),
            "弹序号": idx+1,
            "投放点x(m)": round(r_launch[0],6),
            "投放点y(m)": round(r_launch[1],6),
            "投放点z(m)": round(r_launch[2],6),
            "起爆点x(m)": round(r_det[0],6),
            "起爆点y(m)": round(r_det[1],6),
            "起爆点z(m)": round(r_det[2],6),
            "有效干扰时段(s)": part_str
        })
    df=pd.DataFrame(rows)
    print("\n===== 第3问 优化结果（DE+抛光+进度条） =====")
    print(df.to_string(index=False))
    cov_str="["+ ", ".join([f"[{a:.3f},{b:.3f}]" for a,b in T_cov])+"]"
    print(f"\n整体严格遮蔽时间段：{cov_str}")
    print(f"总覆盖时间 J={union_len(T_cov):.6f} s")


if __name__=="__main__":
    run_q3_progress()