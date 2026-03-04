import numpy as np
from time import strftime
from sklearn.model_selection import train_test_split
from src.aft import compute_AFT_solution
from src.fourier_conversion import (convert_cossin_to_comexp,
                                    convert_comexp_to_cossin)


SAVE_DATA = 0
number_samples = 10000

H = 3        # Number of harmonics
N = 2**6     # Number of time samples
gamma = 0.1  # Nonlinearity parameter


# Sample around phyiscally valid points in 4D space
# Import relevant points from application data
q_coeffs = np.loadtxt('data/nn_input_Duffing.txt', delimiter=',')
a1 = q_coeffs[:, 1]
b1 = q_coeffs[:, 2]
a3 = q_coeffs[:, 5]
b3 = q_coeffs[:, 6]

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


def sample_along_structure(n, noise_scale=1.0, theta_jitter=0.0, r_jitter=0.0):
    """
    noise_scale: Spread on a3,b3 (from residuals), e.g. 50.0
    theta_jitter: small angle jitter (rad), e.g., 2.0
    r_jitter: relative radius jitter, e.g., 0.1
    """

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


all_samples = sample_along_structure(
    number_samples,
    noise_scale=50.0,
    theta_jitter=2.0,
    r_jitter=0.1
)

q_all = []
fnl_all = []
for i in range(number_samples):
    a1 = all_samples[i, 0]
    b1 = all_samples[i, 1]
    a3 = all_samples[i, 2]
    b3 = all_samples[i, 3]
    q_all.append([a1, b1, a3, b3])

    # q_h = a1 * np.cos(t) + b1 * np.sin(t)
    #      + a3 * np.cos(3*t) + b3 * np.sin(3*t)
    q_cs = np.zeros(2 * H + 1)  # fourier coefficients a0, a1, b1, ...
    q_cs[1] = a1
    q_cs[2] = b1
    q_cs[5] = a3
    q_cs[6] = b3
    q_ce = convert_cossin_to_comexp(q_cs)
    fnl_ce = compute_AFT_solution(N, H, q_ce, gamma)
    fnl_cs = convert_comexp_to_cossin(fnl_ce, H)
    fnl_all.append([fnl_cs[1], fnl_cs[2], fnl_cs[5], fnl_cs[6]])


# split data into 60% train, 20% valdation and 20% test
q_tmp, q_test, fnl_tmp, fnl_test = train_test_split(
    q_all, fnl_all, test_size=0.2, random_state=42
)
q_train, q_val, fnl_train, fnl_val = train_test_split(
    q_tmp, fnl_tmp, test_size=0.25, random_state=42
)

if SAVE_DATA:
    current_time = strftime("%Y-%m-%d_%H-%M-%S")
    test_filename = f'duffing_test_data_H{H}_N{N}_{current_time}'
    np.savez(f'data/{test_filename}.npz', q_coeffs=q_test, fnl_coeffs=fnl_test)
    train_filename = f'duffing_train_data_H{H}_N{N}_{current_time}'
    np.savez(f'data/{train_filename}.npz', q_coeffs=q_train,
             fnl_coeffs=fnl_train)
    val_filename = f'duffing_val_data_H{H}_N{N}_{current_time}'
    np.savez(f'data/{val_filename}.npz', q_coeffs=q_val, fnl_coeffs=fnl_val)

print('Generated data:')
print(f'{np.shape(q_all)[0]} samples for ' +
      f'{np.shape(q_all)[1]} input features')
print(f'{np.shape(fnl_all)[0]} samples for corresponding ' +
      f'{np.shape(fnl_all)[1]} output features')
print(f'Data was split into {len(q_train)} training, {len(q_val)} ' +
      f'validation, and {len(q_test)} test samples')
print(f'Data was {"saved" if SAVE_DATA else "NOT saved"}.')
