#长长长长时间！
import math, random
import numpy as np
from tqdm import trange
from dataclasses import dataclass

@dataclass
class PeiZhi:
    g = 9.8
    dandao_v = 300
    yanwu_v = 3.0
    yanwu_r = 10.0
    yanwu_shichang = 20.0
    cyl_r = 7.0
    cyl_h = 10.0
    mubiao_xy = (0.0, 200.0)
    mubiao_z = 0.0
    fy1_start = np.array([17800.0, 0.0, 1800.0])
    dandao_start = np.array([20000.0, 0.0, 2000.0])
    decoy = np.array([0.0, 0.0, 0.0])
    dt = 0.01
    refine_steps = 10
    max_yinqi = 8.0
    max_rel_time = 67.0
    interval_min = 1.0

CFG = PeiZhi()
DANDAO_DIR = (CFG.decoy - CFG.dandao_start)/np.linalg.norm(CFG.decoy - CFG.dandao_start)

# 工具函数
def dandao_pos(t): return CFG.dandao_start + CFG.dandao_v * DANDAO_DIR * t
def fy1_pos(t, theta, v): return CFG.fy1_start + v * np.array([math.cos(theta), math.sin(theta), 0.0]) * t

def calc_yinqi(theta, v, t_rel, tau):
    launch = fy1_pos(t_rel, theta, v)
    vz = -0.5 * CFG.g * tau**2
    det_pt = launch + np.array([v*math.cos(theta), v*math.sin(theta),0.0]) * tau + np.array([0,0,vz])
    return launch, det_pt, t_rel+tau

def get_cloud_center(t, det_pt, t_det):
    if t<t_det or t-t_det>CFG.yanwu_shichang: return None
    return det_pt + np.array([0.0,0.0,-CFG.yanwu_v*(t-t_det)])

# 判定函数
def sample_mubiao_points():
    points=[]
    zs=[CFG.mubiao_z, CFG.mubiao_z+CFG.cyl_h]
    for z in zs:
        for k in range(8):
            angle=k*math.pi/4
            x=CFG.mubiao_xy[0]+CFG.cyl_r*math.cos(angle)
            y=CFG.mubiao_xy[1]+CFG.cyl_r*math.sin(angle)
            points.append(np.array([x,y,z]))
    points.append(np.array([CFG.mubiao_xy[0],CFG.mubiao_xy[1],CFG.mubiao_z]))
    points.append(np.array([CFG.mubiao_xy[0],CFG.mubiao_xy[1],CFG.mubiao_z+CFG.cyl_h]))
    return points

def line_hits_sphere(p0,p1,c,r):
    d=p1-p0; f=p0-c
    a=np.dot(d,d); b=2*np.dot(f,d); c0=np.dot(f,f)-r**2
    disc=b**2-4*a*c0
    if disc<0: return False
    sqrt_disc=math.sqrt(max(0.0,disc))
    t1,t2=(-b-sqrt_disc)/(2*a),(-b+sqrt_disc)/(2*a)
    return 0.0<=t1<=1.0 or 0.0<=t2<=1.0

def merge_intervals(intervals):
    if not intervals: return []
    intervals.sort()
    merged=[list(intervals[0])]
    for start,end in intervals[1:]:
        last=merged[-1]
        if start<=last[1]+1e-9: last[1]=max(last[1],end)
        else: merged.append([start,end])
    return [(a,b) for a,b in merged if b-a>1e-6]

def intersect_all(list_of_sets):
    res=list_of_sets[0]
    for s in list_of_sets[1:]:
        temp=[]
        for a1,b1 in res:
            for a2,b2 in s:
                l,r=max(a1,a2), min(b1,b2)
                if l<r-1e-9: temp.append((l,r))
        res=temp
    return res

def sample_cover_intervals(theta,v,t_rel,tau,idx):
    launch, det_pt, t_det=calc_yinqi(theta,v,t_rel,tau)
    ts=np.arange(t_det,t_det+CFG.yanwu_shichang,CFG.dt)
    points=sample_mubiao_points()
    p=points[idx]
    result=[]
    active=False
    for i in range(len(ts)-1):
        t1,t2=ts[i],ts[i+1]
        m1,m2=dandao_pos(t1), dandao_pos(t2)
        c1,c2=get_cloud_center(t1,det_pt,t_det), get_cloud_center(t2,det_pt,t_det)
        s1=c1 is not None and line_hits_sphere(m1,p,c1,CFG.yanwu_r)
        s2=c2 is not None and line_hits_sphere(m2,p,c2,CFG.yanwu_r)
        if s1 and not active: start=t1; active=True
        if not s2 and active: result.append([start,t2]); active=False
    if active: result.append([start,ts[-1]])
    return merge_intervals(result)

def compute_smoke_effect(theta,v,t_rel,tau):
    all=[sample_cover_intervals(theta,v,t_rel,tau,i) for i in range(10)]
    return [merge_intervals(x) for x in all]

def evaluate_payload_effect(theta,v,trels,taus):
    theta,v,trels,taus=enforce_constraints(theta,v,trels,taus)
    all_points=[[] for _ in range(10)]
    for k in range(3):
        effect=compute_smoke_effect(theta,v,trels[k],taus[k])
        for i in range(10): all_points[i].extend(effect[i])
    for i in range(10): all_points[i]=merge_intervals(all_points[i])
    T=intersect_all(all_points)
    total=sum(b-a for a,b in T)
    return total,T,theta,v,trels,taus

def enforce_constraints(theta,v,trels,taus):
    theta=theta%(2*math.pi)
    v=min(max(v,70.0),140.0)
    trels=sorted(min(max(t,0.0),CFG.max_rel_time) for t in trels)
    for i in [1,2]:
        if trels[i]<trels[i-1]+CFG.interval_min: trels[i]=min(CFG.max_rel_time,trels[i-1]+CFG.interval_min)
    taus=[min(max(x,0.0),CFG.max_yinqi) for x in taus]
    return theta,v,trels,taus

# 差分进化与抛光
class DE:
    def __init__(self,pop=56,gens=300,F=0.7,CR=0.85,seed=2025):
        self.pop,self.gens,self.F,self.CR=pop,gens,F,CR
        random.seed(seed); np.random.seed(seed)

    def rand_candidate(self):
        th=random.uniform(0,2*math.pi)
        v=random.uniform(70,140)
        tr=sorted([random.uniform(0,CFG.max_rel_time) for _ in range(3)])
        for i in [1,2]:
            if tr[i]<tr[i-1]+CFG.interval_min: tr[i]=min(CFG.max_rel_time,tr[i-1]+CFG.interval_min)
        tau=[random.uniform(0,CFG.max_yinqi) for _ in range(3)]
        return np.array([th,v,*sum(zip(tr,tau),())])

    def run(self):
        X=np.vstack([self.rand_candidate() for _ in range(self.pop)])
        fit=np.array([evaluate_payload_effect(*self.unpack(x))[0] for x in X])
        best_idx=int(np.argmax(fit)); best=X[best_idx].copy(); best_fit=fit[best_idx]
        D=X.shape[1]

        for g in trange(self.gens,desc="[DE进化]"):
            for i in range(self.pop):
                a,b,c=random.sample([j for j in range(self.pop) if j!=i],3)
                mutant=X[a]+self.F*(X[b]-X[c])+0.2*(best-X[a])
                cross=np.where(np.random.rand(D)<self.CR,mutant,X[i])
                cross[0]%=2*math.pi; cross[1]=np.clip(cross[1],70,140)
                for k in [2,4,6]: cross[k]=np.clip(cross[k],0,CFG.max_rel_time)
                for k in [3,5,7]: cross[k]=np.clip(cross[k],0,CFG.max_yinqi)
                J=evaluate_payload_effect(*self.unpack(cross))[0]
                if J>=fit[i]:
                    X[i]=cross; fit[i]=J
                    if J>best_fit: best_fit=J; best=cross.copy()
        return best,best_fit

    def unpack(self,x): return x[0],x[1],[x[2],x[4],x[6]],[x[3],x[5],x[7]]

def polish(x,step=[math.radians(4),6,0.6,0.6,0.6,0.6,0.6,0.6],rounds=3):
    def unpack(x): return x[0],x[1],[x[2],x[4],x[6]],[x[3],x[5],x[7]]
    for _ in range(rounds):
        for d in range(8):
            a,b=-step[d],step[d]; phi=(math.sqrt(5)-1)/2
            def f(s):
                x2=x.copy(); x2[d]+=s
                J,*_=evaluate_payload_effect(*unpack(x2))
                return J
            c=b-phi*(b-a); d2=a+phi*(b-a)
            for _ in range(10):
                if f(c)<f(d2): a,b=c,d2
                else: a,b=a,c
                c=b-phi*(b-a); d2=a+phi*(b-a)
            x[d]+=0.5*(a+b); step[d]*=0.6
    return x

if __name__=="__main__":
    de=DE()
    print("[启动] 差分进化初始搜索")
    x0,j0=de.run()
    print(f"[结果] 初步遮蔽时长：{j0:.4f} s")
    print("[优化] 开始局部微调")
    x1=polish(x0.copy())
    j1,T_cov, *_=evaluate_payload_effect(*de.unpack(x1))
    print(f"[完成] 最终遮蔽时长：{j1:.4f} s")
    print("遮蔽时间段：",T_cov)
