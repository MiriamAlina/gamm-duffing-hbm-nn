import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


data_load_date = '2026-02-18_14-04-47'
data = np.load('data/duffing_training_data_H3_N64_'+data_load_date+'.npz')
q_data = data['q_coeffs']
fnl_data = data['fnl_coeffs']
x = [1, 2, 5, 6]
fig, ax = plt.subplots(2, 1, figsize=(5, 6))
ax[0].violinplot(q_data, positions=x, showmeans=False, showextrema=False)
ax[0].hlines(0, 0, 6.5, 'k', linewidth=0.5)
ax[0].set_xlim(0, 6.5)
ax[0].set_xticks(x)
ax[0].set_xticklabels([r'$a_1$', r'$b_1$', r'$a_3$', r'$b_3$'])
ax[0].set_title('Distribution of input features')
ax[1].violinplot(fnl_data, positions=x, showmeans=False, showextrema=False)
ax[1].hlines(0, 0, 6.5, 'k', linewidth=0.5)
ax[1].set_xlim(0, 6.5)
ax[1].set_xticks(x)
ax[1].set_xticklabels([r'$a_1$', r'$b_1$', r'$a_3$', r'$b_3$'])
ax[1].set_title('Distribution of output features')
plt.tight_layout()

fig, ax = plt.subplots(4, 2, figsize=(10, 6))
ax[0, 0].hist([row[0] for row in q_data], bins=20, color='#457B9D', label='a1')
ax[0, 0].legend(loc='upper left')
ax[1, 0].hist([row[1] for row in q_data], bins=20, color='#457B9D', label='b1')
ax[1, 0].legend(loc='upper left')
ax[2, 0].hist([row[2] for row in q_data], bins=20, color='#457B9D', label='a3')
ax[2, 0].legend(loc='upper left')
ax[3, 0].hist([row[3] for row in q_data], bins=20, color='#457B9D', label='b3')
ax[3, 0].legend(loc='upper left')
ax[1, 0].set_ylabel('Number of samples')
ax[0, 0].set_title('Distribution of input features')
ax[0, 1].hist([row[0] for row in fnl_data], bins=20, color='#1D3557',
              label='A1')
ax[0, 1].legend(loc='upper left')
ax[1, 1].hist([row[1] for row in fnl_data], bins=20, color='#1D3557',
              label='B1')
ax[1, 1].legend(loc='upper left')
ax[2, 1].hist([row[2] for row in fnl_data], bins=20, color='#1D3557',
              label='A3')
ax[2, 1].legend(loc='upper left')
ax[3, 1].hist([row[3] for row in fnl_data], bins=20, color='#1D3557',
              label='B3')
ax[3, 1].legend(loc='upper left')
ax[1, 1].set_ylabel('Number of samples')
ax[0, 1].set_title('Distribution of output features')
plt.tight_layout()


q_rel = np.loadtxt('data/nn_input_Duffing.txt', delimiter=',')
data_load_date = '2026-02-18_14-04-47'
data = np.load('data/duffing_training_data_H3_N64_'+data_load_date+'.npz')
q_data = data['q_coeffs']
a1_data = q_data[:, 0]
b1_data = q_data[:, 1]
a3_data = q_data[:, 2]
b3_data = q_data[:, 3]

PLOT_15 = 0
if PLOT_15:
    fig = plt.figure(figsize=(5, 9))

    # --- Original ---
    ax1 = fig.add_subplot(211, projection='3d')
    p1 = ax1.scatter(
        q_rel[:, 1], q_rel[:, 2], q_rel[:, 5],
        c=q_rel[:, 6], s=8, alpha=0.5, cmap='coolwarm'
    )
    p1.set_clim(q_rel[:, 6].min(), q_rel[:, 6].max())
    ax1.set_title("Relevant for application")
    ax1.set_xlabel(r'$a_1$')
    ax1.set_ylabel(r'$b_1$')
    ax1.set_zlabel(r'$a_3$')
    fig.colorbar(p1, ax=ax1, shrink=0.65, label=r'$b_3$', pad=0.15)

    # --- Samples ---
    ax2 = fig.add_subplot(212, projection='3d')
    p2 = ax2.scatter(
        a1_data, b1_data, a3_data,
        c=b3_data, s=8, alpha=0.5, cmap='coolwarm'
    )
    p2.set_clim(q_rel[:, 6].min(), q_rel[:, 6].max())
    ax2.set_title("Training samples")
    ax2.set_xlabel(r'$a_1$')
    ax2.set_ylabel(r'$b_1$')
    ax2.set_zlabel(r'$a_3$')
    fig.colorbar(p2, ax=ax2, shrink=0.65, label=r'$b_3$', pad=0.15)

    # gleiche Achsenlimits fürs faire Vergleichen
    a1_all = np.concatenate([q_rel[:, 1], a1_data])
    b1_all = np.concatenate([q_rel[:, 2], b1_data])
    a3_all = np.concatenate([q_rel[:, 5], a3_data])
    ax1.set_xlim(a1_all.min(), a1_all.max())
    ax2.set_xlim(ax1.get_xlim())
    ax1.set_ylim(b1_all.min(), b1_all.max())
    ax2.set_ylim(ax1.get_ylim())
    ax1.set_zlim(a3_all.min(), a3_all.max())
    ax2.set_zlim(ax1.get_zlim())

    plt.tight_layout()
    plt.savefig('./figures/duffing_training_samples.png', dpi=300,
                bbox_inches='tight')

plt.show()
