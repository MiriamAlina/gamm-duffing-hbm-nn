import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.util import check_folder_structure
from src.plotting import frc_plot, frc_with_inset_plot


###############################################################################
# Check folder structure
###############################################################################
check_folder_structure()


###############################################################################
# Load Reference and Test Results
###############################################################################
ref = pd.read_csv('./results/Duffing_reference_results.csv', header=None)
test = pd.read_csv('./results/Duffing_testing_results.csv', header=None)

diff = ref - test
max_rel_diff = (diff.abs() / ref.abs()).max().max()

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"
tol = 0.5  # 0.5% of the maximum value in the reference results
if max_rel_diff < tol:
    print(GREEN + "Test passed: Results are within the tolerance limit of " +
          str(tol) + "%.")
    print('Maximum difference between reference and test results: ' +
          str(np.round(max_rel_diff, 2)) + '%' + RESET)
else:
    print(RED + "Test failed: Results exceed the tolerance limit of " +
          str(tol) + "%.")
    print('Maximum difference between reference and test results: ' +
          str(np.round(max_rel_diff, 2)) + '%' + RESET)


frc_plot(ref.iloc[0].to_numpy(),
         ref.iloc[1].to_numpy(),
         figure_name='duffing_aft', save_figure=False)
frc_with_inset_plot([ref.iloc[0].to_numpy(), test.iloc[0].to_numpy()],
                    [ref.iloc[1].to_numpy(), test.iloc[1].to_numpy()],
                    figure_name='duffing_aft_nn', save_figure=False)

plt.show()
