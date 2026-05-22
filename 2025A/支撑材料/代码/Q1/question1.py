import numpy as np

# ---------------- 基本常量 ----------------
g = 9.81                  # m/s^2
v_missile = 300.0        # 导弹速度
v_uav = 120.0            # FY1 速度（朝原点）
t_release = 1.5          # 受领后投放时间（s）
t_fuse = 3.6             # 投放后起爆延迟（s）
sink_v = 3.0             # 起爆后竖直下沉速度（m/s）
R_smoke = 10.0           # 有效烟幕半径（m）
t_eff = 20.0             # 起爆后有效时间（s）

# ---------------- 初始状态（题面） ----------------
fake = np.array([0.0, 0.0, 0.0])           # 假目标中心（原点）
M0   = np.array([20000.0, 0.0, 2000.0])    # 导弹 M1 初始
FY1_0= np.array([17800.0, 0.0, 1800.0])    # FY1 初始

# 真目标：圆柱
cyl_center = np.array([0.0, 200.0, 0.0])   # 下底面圆心
cyl_R, cyl_H = 7.0, 10.0

# ---------------- 工具函数 ----------------
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

# 导弹沿直线飞向假目标
u_m = unit(fake - M0)
def missile_pos(t):  # 3D 直线
    return M0 + v_missile * t * u_m

# FY1 等高直线（平面运动：z 不变；方向沿 -x）
def fy1_pos(t):
    return np.array([FY1_0[0] - v_uav * t, FY1_0[1], FY1_0[2]])

# 起爆瞬间球心
def bomb_explode_state():
    # 投放时刻位置
    R_rel = fy1_pos(t_release)
    # 投放到起爆：水平与UAV一致（-x方向），竖直自由落体
    x_e = R_rel[0] - v_uav * t_fuse
    y_e = R_rel[1]
    z_e = R_rel[2] - 0.5 * g * (t_fuse ** 2)
    C_e = np.array([x_e, y_e, z_e])
    t_e = t_release + t_fuse
    return C_e, t_e

C_e, t_e = bomb_explode_state()
t0, t1 = t_e, t_e + t_eff

def smoke_center(t):
    # 起爆后：水平归零，竖直匀速下沉
    if t < t_e or t > t1:
        return None
    return C_e + np.array([0.0, 0.0, -sink_v * (t - t_e)])

# ---------------- 线段-球相交（按二次方程） ----------------
def segment_sphere_intersect(M, P, C, r):
    """
    线段 MP 与球 (C, r) 是否相交：
    X(s) = M + s*(P-M),  s ∈ [0,1]
    |X(s) - C|^2 = r^2  ->  a s^2 + b s + c = 0
      a = |P-M|^2
      b = -2 (C-M)·(P-M)
      c = |C-M|^2 - r^2
    判决：Δ=b^2-4ac ≥ 0 且 根在 [0,1]
    """
    PM = P - M
    CM = C - M
    a = np.dot(PM, PM)
    if a <= 1e-16:  # 退化：M≈P
        return np.dot(CM, CM) <= r*r + 1e-12
    b = -2.0 * np.dot(CM, PM)
    c = np.dot(CM, CM) - r*r
    D = b*b - 4.0*a*c
    if D < 0.0:
        return False
    sqrtD = np.sqrt(D)
    s1 = (-b - sqrtD) / (2.0 * a)
    s2 = (-b + sqrtD) / (2.0 * a)
    return (0.0 <= s1 <= 1.0) or (0.0 <= s2 <= 1.0)

# ---------------- 圆柱“任意点”采样（侧壁+顶/底面） ----------------
def sample_cylinder_rims(n_theta=360):
    """
    仅采“圆柱的顶/底圆的圆周边缘点”：
    - 顶圆：z = cyl_center.z + cyl_H
    - 底圆：z = cyl_center.z
    - 每个圆周按 n_theta 等角采样
    """
    cx, cy, cz = cyl_center
    z_top = cz + cyl_H
    z_bot = cz
    thetas = np.linspace(0.0, 2*np.pi, n_theta, endpoint=False)
    pts = []
    for th in thetas:
        cth, sth = np.cos(th), np.sin(th)
        # 底圆边缘
        pts.append([cx + cyl_R * cth, cy + cyl_R * sth, z_bot])
        # 顶圆边缘
        pts.append([cx + cyl_R * cth, cy + cyl_R * sth, z_top])
    return np.array(pts, dtype=float)


# 采样密度（可根据精度/速度调节；当前设置可稳定得到 ~1.4 s）
target_points = sample_cylinder_rims(n_theta=3600)


# ---------------- 单时刻“完全遮盖”判定 ----------------
def fully_covered_at_time(t, block=4096):
    C = smoke_center(t)
    if C is None:
        return False
    M = missile_pos(t)

    P = target_points
    n = P.shape[0]
    # 分块向量化，避免一次性内存过大
    for i in range(0, n, block):
        Pi = P[i:i+block]
        # 向量化批量判定
        PM = Pi - M
        CM = C - M
        a = np.einsum('ij,ij->i', PM, PM)
        # 退化点（极少）
        mask = a <= 1e-16
        if np.any(mask):
            d2 = np.sum(CM*CM)
            if d2 > R_smoke*R_smoke + 1e-12:
                return False
            PM = PM[~mask]; a = a[~mask]
            if PM.shape[0] == 0:
                continue
        b = -2.0 * np.dot(CM, PM.T)      # (k,)
        c = np.sum(CM*CM) - R_smoke**2   # 标量
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

# ---------------- 统计遮蔽时长（时间扫描 + 边界二分） ----------------
def cover_time(dt=0.01, refine_steps=24):
    times = np.arange(t0, t1 + 1e-12, dt)
    covered = np.array([fully_covered_at_time(t) for t in times], dtype=bool)

    total = 0.0
    intervals = []
    i, n = 0, len(times)

    while i < n-1:
        # 找进入边界
        while i < n-1 and not (not covered[i] and covered[i+1]):
            i += 1
        if i >= n-1:
            break
        # 进入时刻二分
        L, R = times[i], times[i+1]
        for _ in range(refine_steps):
            mid = 0.5*(L+R)
            if fully_covered_at_time(mid):
                R = mid
            else:
                L = mid
        t_in = 0.5*(L+R)

        # 扫描到退出
        j = i+1
        while j < n and covered[j]:
            j += 1
        # 退出时刻二分
        L = times[j-1]
        R = times[j] if j < n else min(times[-1]+dt, t1)
        for _ in range(refine_steps):
            mid = 0.5*(L+R)
            if fully_covered_at_time(mid):
                L = mid
            else:
                R = mid
        t_out = 0.5*(L+R)

        total += max(0.0, t_out - t_in)
        intervals.append((t_in, t_out))
        i = j

    return total, intervals

# ---------------- 运行 ----------------
if __name__ == "__main__":
    print(f"起爆时刻 t_e = {t_e:.3f} s")
    print(f"起爆球心 C_e = {C_e}")
    print(f"有效时间窗: [{t0:.3f}, {t1:.3f}] s")

    t_cov, segs = cover_time(dt=0.002, refine_steps=30)
    print(f"有效遮蔽时长 ≈ {t_cov:.4f} s")
    for a,b in segs:
        print(f"遮蔽区间: [{a:.6f}, {b:.6f}] s")
