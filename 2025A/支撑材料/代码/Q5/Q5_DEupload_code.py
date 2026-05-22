#长长长长！2小时以上
import numpy as np
from math import cos, sin, pi, sqrt, degrees
import random

np.random.seed(2025)
random.seed(2025)


class Shijiao:
    mubiao = np.array([0.0, 200.0, 0.0])
    cyl_r = 7.0
    cyl_h = 10.0
    yanwu_r = 10.0
    yanwu_v = 3.0
    yanwu_life = 20.0
    g = 9.8
    fy1_init = np.array([0.0,0.0,0.0])
    daodan = np.array([
        [20000,0,2000],
        [19000,600,2100],
        [18000,-600,1900]
    ])
    daodan_v = 300.0
    fy1_start = np.array([6000.0,-3000.0,700.0])
    dt = 0.01
    refine = 10
    t_max_det = 35.0
    t_max_throw = 28.0
    t_min_interval = 1.0

def unit_vector(v): 
    n = np.linalg.norm(v); return v/n if n!=0 else v

def daodan_pos(t, m0):
    dir_vec = unit_vector(Shijiao.fy1_init - m0)
    return m0 + Shijiao.daodan_v * dir_vec * t

def fy1_pos(t, theta, v):
    dir_xy = np.array([cos(theta), sin(theta), 0.0])
    return Shijiao.fy1_start + v*dir_xy*t

def kaihuo(theta, v, t_tou, tau):
    r0 = fy1_pos(t_tou, theta, v)
    accel = np.array([0.0,0.0,-Shijiao.g])
    r_det = r0 + np.array([v*cos(theta),v*sin(theta),0.0])*tau + 0.5*accel*(tau**2)
    return r0, r_det, t_tou+tau

def yanwu_pos(t, r_det, t_det):
    if t < t_det or t - t_det > Shijiao.yanwu_life: return None
    dt = t - t_det
    return r_det + np.array([0.0,0.0,-Shijiao.yanwu_v*dt])

def cyl_pts():
    n_rad, n_h = 8, 2
    pts=[]
    zs = np.linspace(0,Shijiao.cyl_h,n_h)
    for z in zs:
        for i in range(n_rad):
            th = 2*pi*i/n_rad
            x = Shijiao.mubiao[0]+Shijiao.cyl_r*cos(th)
            y = Shijiao.mubiao[1]+Shijiao.cyl_r*sin(th)
            pts.append([x,y,z])
    pts.append([Shijiao.mubiao[0],Shijiao.mubiao[1],0])
    pts.append([Shijiao.mubiao[0],Shijiao.mubiao[1],Shijiao.cyl_h])
    return np.array(pts)

def seg_hits_sphere(p0,p1,c,R):
    d = p1-p0; f = p0-c
    a = np.dot(d,d); b=2*np.dot(f,d); c0=np.dot(f,f)-R**2
    disc = b*b-4*a*c0
    if disc<0: return False
    s1 = (-b - sqrt(max(0,disc)))/(2*a)
    s2 = (-b + sqrt(max(0,disc)))/(2*a)
    return 0<=s1<=1 or 0<=s2<=1

def merge_intervals(intervals):
    if len(intervals)==0: return np.zeros((0,2))
    intervals=sorted(intervals,key=lambda x:x[0])
    merged=[intervals[0]]
    for a,b in intervals[1:]:
        la,lb=merged[-1]
        if a<=lb+1e-9: merged[-1][1]=max(lb,b)
        else: merged.append([a,b])
    return np.array(merged)

def intersect_two(A,B):
    i=j=0; out=[]
    while i<len(A) and j<len(B):
        a1,b1=A[i]; a2,b2=B[j]
        lo,hi=max(a1,a2), min(b1,b2)
        if lo<hi: out.append([lo,hi])
        if b1<b2: i+=1
        else: j+=1
    return np.array(out)

def intersect_all(lst):
    out=lst[0]
    for u in lst[1:]:
        out=intersect_two(out,u)
        if len(out)==0: break
    return out

def total_len(U):
    return np.sum(U[:,1]-U[:,0]) if len(U)>0 else 0.0

def clamp_params(theta,v,trels,taus):
    theta=theta%(2*pi)
    v=np.clip(v,70,140)
    trels=np.clip(trels,0,Shijiao.t_max_throw); trels=np.sort(trels)
    for i in range(1,3):
        if trels[i]<trels[i-1]+Shijiao.t_min_interval:
            trels[i]=min(Shijiao.t_max_throw, trels[i-1]+Shijiao.t_min_interval)
    taus=np.maximum(taus,0)
    for k in range(3):
        t_det=trels[k]+taus[k]
        if t_det>Shijiao.t_max_det:
            over=t_det-Shijiao.t_max_det
            dec=min(taus[k],over)
            taus[k]-=dec; over-=dec
            if over>1e-9: trels[k]=max(0,trels[k]-over)
    return theta,v,trels,taus

def intervals_point(theta,v,t_tou,tau,idx,m0):
    r0,r_det,t_det=kaihuo(theta,v,t_tou,tau)
    if not (0<=t_det<=Shijiao.t_max_det): return np.zeros((0,2))
    ts=np.arange(t_det,t_det+Shijiao.yanwu_life+1e-12,Shijiao.dt)
    pts=cyl_pts()
    def hit(t):
        c=yanwu_pos(t,r_det,t_det)
        return seg_hits_sphere(daodan_pos(t,m0),pts[idx],c,Shijiao.yanwu_r) if c is not None else False
    out=[]; prev=hit(ts[0])
    if prev: out.append([ts[0],np.nan])
    for t in ts[1:]:
        s=hit(t)
        if s!=prev:
            lo,hi=ts[0],t
            for _ in range(Shijiao.refine): mid=0.5*(lo+hi); sm=hit(mid); lo=mid if sm==prev else lo; hi=mid if sm!=prev else hi
            tsw=0.5*(lo+hi)
            if not prev and s: out.append([tsw,np.nan])
            elif prev and not s and len(out)>0 and np.isnan(out[-1][1]): out[-1][1]=tsw
        prev=s
    if prev and len(out)>0 and np.isnan(out[-1][1]): out[-1][1]=ts[-1]
    out=[iv for iv in out if iv[1]-iv[0]>1e-9]
    return merge_intervals(out)

def bomb_for_missile(theta,v,t_tou,tau,m0):
    per=[intervals_point(theta,v,t_tou,tau,i,m0) for i in range(6)]
    return {'r_launch':kaihuo(theta,v,t_tou,tau)[0],
            'r_det':kaihuo(theta,v,t_tou,tau)[1],
            't_det':kaihuo(theta,v,t_tou,tau)[2],
            'per_point':per,
            'union':merge_intervals(np.vstack(per))}

def strict_union_3bombs(B1,B2,B3):
    unions=[merge_intervals(np.vstack([B1['per_point'][i],B2['per_point'][i],B3['per_point'][i]])) for i in range(6)]
    cov=intersect_all(unions)
    return cov,total_len(cov)

# DE 优化 
def rand_individual():
    theta=2*pi*np.random.rand(); v=70+70*np.random.rand()
    trels=np.sort(Shijiao.t_max_throw*np.random.rand(3))
    for i in [1,2]: trels[i]=max(trels[i],trels[i-1]+Shijiao.t_min_interval)
    taus=8*np.random.rand(3)
    return np.hstack([theta,v,trels[0],taus[0],trels[1],taus[1],trels[2],taus[2]])

def objective(theta,v,trels,taus):
    theta,v,trels,taus=clamp_params(theta,v,np.array(trels),np.array(taus))
    total=0.0
    for m0 in Shijiao.daodan:
        bombs=[bomb_for_missile(theta,v,trels[k],taus[k],m0) for k in range(3)]
        _,J=strict_union_3bombs(bombs[0],bombs[1],bombs[2])
        total+=J
    return total

def eval_x(x): return objective(x[0],x[1],[x[2],x[4],x[6]],[x[3],x[5],x[7]])

def run_de():
    pop,gens,F,CR=56,250,0.7,0.9
    X=np.array([rand_individual() for _ in range(pop)])
    fit=np.array([eval_x(X[i,:]) for i in range(pop)])
    best_idx=np.argmax(fit); best_fit=fit[best_idx]; best=X[best_idx,:]
    print("[DE开始][三导弹] 粗解计算中...")
    for g in range(gens):
        for i in range(pop):
            idxs=[j for j in range(pop) if j!=i]; a,b,c=np.random.choice(idxs,3,replace=False)
            mutant=X[a]+F*(X[b]-X[c])+0.2*(best-X[a])
            cross=np.array([mutant[j] if np.random.rand()<CR or j==np.random.randint(len(mutant)) else X[i,j] for j in range(len(mutant))])
            cross[0]=cross[0]%(2*pi); cross[1]=np.clip(cross[1],70,140)
            for idx in [2,4,6]: cross[idx]=np.clip(cross[idx],0,Shijiao.t_max_throw)
            for idx in [3,5,7]: cross[idx]=np.clip(cross[idx],0,8)
            val=eval_x(cross)
            if val>=fit[i]: X[i,:]=cross; fit[i]=val; 
            if val>best_fit: best_fit=val; best=cross
        print(f"[DE][三导弹] gen={g+1}/{gens}, best J_total={best_fit:.6f}")
    print(f"\n[DE结束][三导弹] 粗解 J_total={best_fit:.6f}")
    return best

if __name__=="__main__":
    best=run_de()
    print("优化完成，最佳向量：", best)
