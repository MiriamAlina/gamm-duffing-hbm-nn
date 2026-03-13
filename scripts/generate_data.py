import numpy as np
from time import strftime
from sklearn.model_selection import train_test_split
from src.aft import compute_AFT_solution
from src.fourier_conversion import (convert_cossin_to_comexp,
                                    convert_comexp_to_cossin)
from src.trajectory_sampling import sample_along_trajectory


SAVE_DATA = True
number_samples = 10000

H = 3        # Number of harmonics
N = 2**6     # Number of time samples
gamma = 0.1  # Nonlinearity parameter

# Sample inputs around phyiscally valid points in 4D space
# Import relevant points from application data
q_coeffs = np.loadtxt('data/input_frc_aft.txt', delimiter=',')
all_samples = sample_along_trajectory(
    q_coeffs,
    number_samples,
    noise_scale=50.0,
    theta_jitter=2.0,
    r_jitter=0.1
)

# Obtain outputs from AFT
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


# Split data into 60% train, 20% valdation and 20% test
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
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
if SAVE_DATA:
    print(f"{GREEN}Data was saved.{RESET}")
else:
    print(f"{RED}Data was NOT saved. To save, set SAVE_DATA = True.{RESET}")
