import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------
# 1) Raw data  (K_est, avg_dis_p, RankMe, R)
# ---------------------------------------------
data = np.array([
    [626, 3.36e+01, 49.61, 1.54e+00],
    [829, 1.91e+01, 45.17, 1.01e+00],
    [408, 1.30e+01, 38.81, 7.43e-01],
    [482, 6.49e+00, 23.14, 4.93e-01],
])

K_est, avg_dis_p, RankMe, R = data.T   # unpack columns

# ---------------------------------------------
# 2) Quality computation
# ---------------------------------------------
N_vis = 100000        # <-- SET the number of visible points here
eps   = 1e-10     # small constant to avoid division by zero
quality = np.zeros(len(K_est))
for i in range(len(K_est)):
    try:
        quality[i] = float(np.log(max(N_vis, 2)) / np.log(1 + 2 * R[i] * K_est[i] / (avg_dis_p[i] + eps))) if avg_dis_p[i] > eps else float('nan')
    except ZeroDivisionError:
        quality[i] = float('nan')

# ---------------------------------------------
# 3) Plot RankMe vs quality
# ---------------------------------------------
plt.figure(figsize=(6, 4))
plt.scatter(RankMe, quality, s=80, color='dodgerblue')
plt.xlabel('RankMe')
plt.ylabel('Estimated Dimension')
plt.title('RankMe vs Estimated Dimension')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('rankme_vs_est_dim.png')