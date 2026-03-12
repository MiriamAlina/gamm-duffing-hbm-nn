import numpy as np
import pandas as pd
from src.plotting import frf_plot, frf_with_inset_plot


###############################################################################
# Analytical Solution for Comparison
###############################################################################
zeta = 0.05     # damping ratio
gamma = 0.1     # nonlinearity coefficient
P = 0.18        # forcing amplitude
a_min = .1      # minimum amplitude for analytical solution
a_max = 3       # maximum amplitude for analytical solution
a_ana = np.linspace(a_min, a_max, 50)  # amplitudes for analytical solution
# For each amplitude, determine the two associated excitation frequencies.
# Only the real-valued solutions exist.
Om_ana = np.zeros((len(a_ana), 2), dtype=complex)
for i, a in enumerate(a_ana):
    prefactor = 1 - zeta**2 / 2 + 3 * gamma * a**2 / 4
    inner = P**2 / a**2 + zeta**4 / 4 - zeta**2 - \
        3 * zeta**2 * gamma * a**2 / 4
    Om_ana[i, 0] = np.sqrt(prefactor + np.sqrt(inner))
    Om_ana[i, 1] = np.sqrt(prefactor - np.sqrt(inner))
valid_ana = (np.isreal(Om_ana[:, 0])) & (np.isreal(Om_ana[:, 1]))

###############################################################################
# Load Reference and Test Results
###############################################################################
ref = pd.read_csv('./results/Duffing_reference_results.csv', header=None)
test = pd.read_csv('./results/Duffing_testing_results.csv', header=None)

diff = ref - test
max_diff = diff.abs().max().max()

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"
tol = 1e-6
if max_diff < tol:
    print(GREEN + "Test passed: Results are within the tolerance limit.")
    print('Maximum difference between reference and test results: ' +
          str(max_diff) + RESET)
else:
    print(RED + "Test failed: Results exceed the tolerance limit.")
    print('Maximum difference between reference and test results: ' +
          str(max_diff) + RESET)


frf_plot(ref.iloc[0].to_numpy(),
         ref.iloc[1].to_numpy(),
         figure_name='duffing_aft')
frf_with_inset_plot([ref.iloc[0].to_numpy(), test.iloc[0].to_numpy()],
                    [ref.iloc[1].to_numpy(), test.iloc[1].to_numpy()],
                    figure_name='duffing_aft_nn')
