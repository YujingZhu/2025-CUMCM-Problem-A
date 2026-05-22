import numpy as np

# ============================================================
# 问题1·严格口径（“所有线段均被球阻挡”）——有效遮蔽时长计算
# 规则：
# 1) 判定“完全遮蔽”：目标上任一点 P，线段 MP 与球(C,R)有交点；仅当对所有 P 都成立，该时刻视为完全遮蔽
# 2) 投放后：弹体水平速度 = UAV 速度向量；竖直方向自由落体
#    起爆后：球心水平速度=0，竖直以 sink_v 匀速下沉；有效 t_eff 秒；半径 R_smoke
# 3) 飞行器等高直线飞（z 不变，XY 平面匀速直线）
# 4) 坐标：以“假目标”为原点；真目标圆柱：下底圆心(0,200,0)，半径7，高10
# ============================================================

# --------------- 参数区（只需改这里） ---------------
# 最优参数（按你给出的结果）
theta1   = 0.1317      # UAV 航向（rad）, 在 XY 平面，0 表示 +x，逆时针为正
v_uav    = 106.69      # UAV 速度 (m/s)
t_release= 0.0         # 投放延迟 (s)
t_fuse   = 0.864       # 引信时间 (s)

# 物理/几何常量
g = 9.8                 # 重力 (m/s^2)
v_missile = 300.0       # 导弹速度 (m/s)
sink_v = 3.0            # 烟幕下沉速度 (m/s)
R_smoke = 10.0          # 烟幕半径 (m)
t_eff = 20.0            # 起爆后有效时间 (s)

# 参与方初始状态
fake = np.array([0.0, 0.0, 0.0])              # 假目标（原点）
M0   = np.array([20000.0, 0.0, 2000.0])       # 导弹 M1 初始
FY1_0= np.array([17800.0, 0.0, 1800.0])       # FY1 初始

# 真目标：圆柱（下底圆心、半径、高度）
cyl_center = np.array([0.0, 200.0, 0.0])
cyl_R, cyl_H = 7.0, 10.0

# 采样与数值控制（可调）
N_THETA_RIM = 900      # 顶/底圆周等角点数（每个圆周 N_THETA_RIM 个点，总点数 2*N_THETA_RIM）
DT_SCAN     = 0.02     # 时间粗扫描步长 (s)
N_REFINE    = 20       # 进入/退出边界二分次数

# ============================================================

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

# 导弹沿直线飞向原点
u_m = unit(fake - M0)
def missile_pos(t):
    return M0 + v_missile * t * u_m

# FY1 等高直线，按给定航向 theta1 匀速前进
u_dir = np.array([np.cos(theta1), np.sin(theta1), 0.0])
def fy1_pos(t):
    return FY1_0 + v_uav * t * u_dir

# 投放→起爆：水平随 UAV 速度向量漂移，竖直自由落体
def bomb_explode_state():
    R_rel = fy1_pos(t_release)              # 投放时刻 UAV 位置
    C_e = R_rel + v_uav * t_fuse * u_dir    # 起爆前水平漂移
    C_e = C_e.copy()
    C_e[2] = C_e[2] - 0.5 * g * (t_fuse ** 2)  # 自由落体
    t_e = t_release + t_fuse
    return C_e, t_e

C_e, t_e = bomb_explode_state()
t0, t1 = t_e, t_e + t_eff

def smoke_center(t):
    if t < t_e or t > t1:
        return None
    return C_e + np.array([0.0, 0.0, -sink_v * (t - t_e)])

# 顶/底圆周采样（严格口径下，“所有线段”≈“圆周极端”判定更紧）
def sample_cylinder_rims(n_theta=360):
    cx, cy, cz = cyl_center
    z_top, z_bot = cz + cyl_H, cz
    thetas = np.linspace(0.0, 2*np.pi, n_theta, endpoint=False)
    P = np.zeros((2*n_theta, 3))
    k = 0
    for th in thetas:
        cth, sth = np.cos(th), np.sin(th)
        P[k] = [cx + cyl_R*cth, cy + cyl_R*sth, z_bot]; k += 1
        P[k] = [cx + cyl_R*cth, cy + cyl_R*sth, z_top]; k += 1
    return P

target_points = sample_cylinder_rims(N_THETA_RIM)  # 2*N_THETA_RIM 个采样点

# 线段-球相交（线段 MP 与球(C,r)）
def segment_sphere_intersect(M, P, C, r):
    PM = P - M
    CM = C - M
    a = np.dot(PM, PM)
    if a <= 1e-16:  # M≈P 退化
        return np.dot(CM, CM) <= r*r + 1e-12
    b = -2.0 * np.dot(CM, PM)
    c = np.dot(CM, CM) - r*r
    D = b*b - 4.0*a*c
    if D < 0.0:
        return False
    sqrtD = np.sqrt(max(D, 0.0))
    s1 = (-b - sqrtD) / (2.0*a)
    s2 = (-b + sqrtD) / (2.0*a)
    return (0.0 <= s1 <= 1.0) or (0.0 <= s2 <= 1.0)

# 单时刻“完全遮蔽”判定：所有采样点均满足线段-球相交
def fully_covered_at_time(t, block=2048):
    C = smoke_center(t)
    if C is None:
        return False
    M = missile_pos(t)
    P = target_points
    n = P.shape[0]

    # 预先算一次，便于退化分支
    CM = C - M
    CM2 = np.dot(CM, CM)
    r2  = R_smoke**2

    for i in range(0, n, block):
        Pi = P[i:i+block]
        PM = Pi - M
        a = np.einsum('ij,ij->i', PM, PM)
        mask = a <= 1e-16
        if np.any(mask):
            # 若存在 M≈P 的点，需直接检查球心到 M 的距离
            if CM2 > r2 + 1e-12:
                return False
            PM = PM[~mask]; a = a[~mask]
            if PM.shape[0] == 0:
                continue
        b = -2.0 * np.dot(CM, PM.T)   # (k,)
        c = CM2 - r2                  # 标量
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

# 扫描 + 二分精修边界，累计完全遮蔽时长
def cover_time(dt=DT_SCAN, refine_steps=N_REFINE):
    times = np.arange(t0, t1 + 1e-12, dt)
    covered = np.array([fully_covered_at_time(t) for t in times], dtype=bool)

    total = 0.0
    intervals = []
    i, n = 0, len(times)

    while i < n-1:
        # 寻找进入边界（False -> True）
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

        # 前进到退出边界
        j = i+1
        while j < n and covered[j]:
            j += 1

        # 退出时刻二分（True -> False）
        L = times[j-1]
        R = times[j] if j < n else min(times[-1] + dt, t1)
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
    print(f"起爆时刻 t_e = {t_e:.6f} s")
    print(f"起爆球心 C_e = [{C_e[0]:.6f}, {C_e[1]:.6f}, {C_e[2]:.6f}]")
    print(f"有效窗口: [{t0:.6f}, {t1:.6f}] s")

    t_cov, segs = cover_time()
    print(f"\n严格口径·完全遮蔽时长 ≈ {t_cov:.4f} s")
    for a, b in segs:
        print(f"遮蔽区间: [{a:.6f}, {b:.6f}] s")

    # —— 稳定性小检查（可选）：改不同采样密度对比 —— #
    # for ntheta in [720, 900, 1200]:
    #     target_points = sample_cylinder_rims(ntheta)
    #     t_cov2, _ = cover_time(dt=DT_SCAN, refine_steps=N_REFINE)
    #     print(f"(采样 {ntheta:>4d}) 遮蔽时长 ≈ {t_cov2:.4f} s")
