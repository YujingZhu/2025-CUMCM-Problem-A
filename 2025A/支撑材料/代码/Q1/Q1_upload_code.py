import numpy as np

# 非常常规的根据已知参数做题
# 非常常见的物理学的解法
gz = 9.8                 
v_dandao = 300.0        
v_wurenji = 120.0            
t_toufang = 1.5          
t_yinqi = 3.6             
v_xiazhen = 3.0             
R_yanwu = 10.0           
t_xiaoguo = 20.0             

ling = np.array([0.0, 0.0, 0.0])          
M0 = np.array([20000.0, 0.0, 2000.0])    
FY1_0 = np.array([17800.0, 0.0, 1800.0])    

# 真目标：圆柱，半径7，高度10
cyl_zhongxin = np.array([0.0, 200.0, 0.0])   
cyl_banjing, cyl_gao = 7.0, 10.0



def danwei(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

u_dandao = danwei(ling - M0)

def weizhi_dandao(t):  
    return M0 + v_dandao * t * u_dandao


def weizhi_wurenji(t):
    return np.array([FY1_0[0] - v_wurenji * t, FY1_0[1], FY1_0[2]])


def baozha():
    pos_toufang = weizhi_wurenji(t_toufang)
    x_e = pos_toufang[0] - v_wurenji * t_yinqi
    y_e = pos_toufang[1]
    z_e = pos_toufang[2] - 0.5 * gz * (t_yinqi ** 2)
    C_e = np.array([x_e, y_e, z_e])
    t_e = t_toufang + t_yinqi
    return C_e, t_e

C_e, t_e = baozha()
t0, t1 = t_e, t_e + t_xiaoguo


def yanwu_zhongxin(t):
    if t < t_e or t > t1:
        return None
    return C_e + np.array([0.0, 0.0, -v_xiazhen * (t - t_e)])

# 采样圆柱上下底面
def caifen_cyl(n_theta=360):
    cx, cy, cz = cyl_zhongxin
    z_top = cz + cyl_gao
    z_bot = cz
    thetas = np.linspace(0.0, 2*np.pi, n_theta, endpoint=False)
    pts = []
    for th in thetas:
        cth, sth = np.cos(th), np.sin(th)
        pts.append([cx + cyl_banjing * cth, cy + cyl_banjing * sth, z_bot])
        pts.append([cx + cyl_banjing * cth, cy + cyl_banjing * sth, z_top])
    return np.array(pts, dtype=float)

target_points = caifen_cyl(n_theta=3600)

# 判断某时刻是否全覆盖
def quanfeng_fuzai(t, block=4096):
    C = yanwu_zhongxin(t)
    if C is None:
        return False
    M = weizhi_dandao(t)
    P = target_points
    n = P.shape[0]

    for i in range(0, n, block):
        Pi = P[i:i+block]
        PM = Pi - M
        CM = C - M
        a = np.einsum('ij,ij->i', PM, PM)
        mask = a <= 1e-16
        if np.any(mask):
            d2 = np.sum(CM*CM)
            if d2 > R_yanwu*R_yanwu + 1e-12:
                return False
            PM = PM[~mask]; a = a[~mask]
            if PM.shape[0] == 0:
                continue
        b = -2.0 * np.dot(CM, PM.T)
        c = np.sum(CM*CM) - R_yanwu**2
        D = b*b - 4.0*a*c
        if np.any(D < 0.0):
            return False
        sqrtD = np.sqrt(np.maximum(D, 0.0))
        s1 = (-b - sqrtD) / (2.0*a)
        s2 = (-b + sqrtD) / (2.0*a)
        hit = ((s1 >= 0.0) & (s1 <= 1.0)) | ((s2 >= 0.0) & (s2 <= 1.0))
        if not np.all(hit):
            return False
    return True

# 计算总覆盖时间和时间段
def zong_fuzai_shijian(dt=0.01, refine_steps=24):
    times = np.arange(t0, t1 + 1e-12, dt)
    covered = np.array([quanfeng_fuzai(t) for t in times], dtype=bool)
    total = 0.0
    intervals = []
    i, n = 0, len(times)

    while i < n-1:
        while i < n-1 and not (not covered[i] and covered[i+1]):
            i += 1
        if i >= n-1:
            break
        L, R = times[i], times[i+1]
        for _ in range(refine_steps):
            mid = 0.5*(L+R)
            if quanfeng_fuzai(mid):
                R = mid
            else:
                L = mid
        t_in = 0.5*(L+R)
        j = i+1
        while j < n and covered[j]:
            j += 1
        L = times[j-1]
        R = times[j] if j < n else min(times[-1]+dt, t1)
        for _ in range(refine_steps):
            mid = 0.5*(L+R)
            if quanfeng_fuzai(mid):
                L = mid
            else:
                R = mid
        t_out = 0.5*(L+R)
        total += max(0.0, t_out - t_in)
        intervals.append((t_in, t_out))
        i = j
    return total, intervals

if __name__ == "__main__":
    print(f"起爆时刻 t_e = {t_e:.3f} s")
    print(f"起爆球心 C_e = {C_e}")
    print(f"有效时间窗: [{t0:.3f}, {t1:.3f}] s")

    t_cov, segs = zong_fuzai_shijian(dt=0.002, refine_steps=30)
    print(f"有效遮蔽时长 ≈ {t_cov:.4f} s")
    for a,b in segs:
        print(f"遮蔽区间: [{a:.6f}, {b:.6f}] s")
