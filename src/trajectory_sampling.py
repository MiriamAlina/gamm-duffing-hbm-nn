import numpy as np


def sample_along_trajectory(trajectory_data, n, noise_scale=1.0,
                            theta_jitter=0.0, r_jitter=0.0):
    """
    noise_scale: Spread on a3,b3 (from residuals), e.g. 50.0
    theta_jitter: small angle jitter (rad), e.g., 2.0
    r_jitter: relative radius jitter, e.g., 0.1
    """

    a1 = trajectory_data[:, 1]
    b1 = trajectory_data[:, 2]
    a3 = trajectory_data[:, 5]
    b3 = trajectory_data[:, 6]

    # a1 vs b1 form a circle with center at (c_a1, c_b1)
    c_a1 = np.mean(a1)
    c_b1 = np.mean(b1)

    # Paramtereization through latent parameters (theta, r)
    theta = np.arctan2(b1 - c_b1, a1 - c_a1)  # [-pi, pi]
    r = np.sqrt((a1 - c_a1)**2 + (b1 - c_b1)**2)

    # Feature matrix
    K = 2
    cols = [np.ones_like(theta), r, r**2]
    for k in range(1, K+1):
        cols += [np.cos(k*theta), np.sin(k*theta),
                 r*np.cos(k*theta), r*np.sin(k*theta)]
    Phi = np.column_stack(cols)

    # Least squares fit for a3 and b3
    w_a3, *_ = np.linalg.lstsq(Phi, a3, rcond=None)
    w_b3, *_ = np.linalg.lstsq(Phi, b3, rcond=None)

    # Residuals as a measure of spread
    a3_fit = Phi @ w_a3
    b3_fit = Phi @ w_b3
    sig_a3 = np.std(a3 - a3_fit)
    sig_b3 = np.std(b3 - b3_fit)

    # 1) Sample (theta, r) from application relevant points (Bootstrap)
    idx = np.random.randint(0, len(theta), size=n)
    th = theta[idx].copy()
    rr = r[idx].copy()

    # small jitter along/across the curve
    if theta_jitter > 0:
        th += theta_jitter * np.random.randn(n)
    if r_jitter > 0:
        rr *= (1.0 + r_jitter * np.random.randn(n))

    # 2) Reconstruct a1,b1 from (r,theta)
    a1_s = c_a1 + rr * np.cos(th)
    b1_s = c_b1 + rr * np.sin(th)

    # 3) Build features for (r,theta)
    cols_s = [np.ones_like(th), rr, rr**2]
    for k in range(1, K+1):
        cols_s += [np.cos(k*th), np.sin(k*th),
                   rr*np.cos(k*th), rr*np.sin(k*th)]
    Phi_s = np.column_stack(cols_s)

    # 4) a3,b3 from model + noise
    a3_s = Phi_s @ w_a3 + noise_scale * sig_a3 * np.random.randn(n)
    b3_s = Phi_s @ w_b3 + noise_scale * sig_b3 * np.random.randn(n)

    return np.column_stack([a1_s, b1_s, a3_s, b3_s])
