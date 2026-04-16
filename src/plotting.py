import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


input_labels = [r"$a_1$", r"$b_1$", r"$a_3$", r"$b_3$"]
output_labels = [r"$A_1$", r"$B_1$", r"$A_3$", r"$B_3$"]
input_symbols = [r"a_1", r"b_1", r"a_3", r"b_3"]
output_symbols = [r"A_1", r"B_1", r"A_3", r"B_3"]
two_colors_set = ['#1D3557', '#e63946']
three_colors_set = ['#A8DADC', '#1D3557', '#e63946']
four_colors_set = ['#1D3557', '#008b9a', '#f19699', '#e63946']


def error_metrics_spider_plot(metrics_dict,
                              normalized_metrics_dict,
                              figure_name, save_figure=False):
    """
    Create a spider plot comparing error metrics before and after
    normalization.
    Parameters
    ----------
    metrics_dict : dict
        Dictionary of error metrics before normalization.
    normalized_metrics_dict : dict
        Dictionary of error metrics after normalization.
    figure_name : str
        Name of the figure file to save.
    save_figure : bool, optional
        Whether to save the figure (default is False).
    """
    labels = list(metrics_dict.keys())
    values = [metrics_dict[k] for k in labels]
    values_norm = [normalized_metrics_dict[k] for k in labels]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    values = np.r_[values, values[0]]
    values_norm = np.r_[values_norm, values_norm[0]]
    angles = np.r_[angles, angles[0]]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"projection": "polar"})
    ax.set_yscale("log")
    ax.plot(angles, values, marker="o", label="raw", color=two_colors_set[0])
    ax.fill(angles, values, alpha=0.2, color=two_colors_set[0])
    ax.plot(angles, values_norm, marker="o", label="normalized",
            color=two_colors_set[1])
    ax.fill(angles, values_norm, alpha=0.2, color=two_colors_set[1])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', pad=12)
    ax.set_ylim(min(values)/1.5, max(values)*3)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.2), ncol=2)
    plt.tight_layout()
    if save_figure:
        plt.savefig(f'figures/{figure_name}.svg', bbox_inches='tight')


def individual_error_metrics_bar_plot(metrics_dict,
                                      normalized_metrics_dict,
                                      figure_name, save_figure=False):
    """
    Create a bar plot comparing error metrics of individual outputs before and
    after normalization.
    Parameters
    ----------
    metrics_dict : dict
        Dictionary of individual error metrics before normalization.
    normalized_metrics_dict : dict
        Dictionary of individual error metrics after normalization.
    figure_name : str
        Name of the figure file to save.
    save_figure : bool, optional
        Whether to save the figure (default is False).
    """
    metrics = metrics_dict.keys()
    raw = np.vstack([metrics_dict[k] for k in metrics])
    norm = np.vstack([normalized_metrics_dict[k] for k in metrics])
    x = np.arange(len(metrics))
    w = 0.18
    fig, ax = plt.subplots(1, 2, figsize=(7, 4), sharey=True)
    for k, (title, data) in enumerate(zip(["raw", "normalized"], [raw, norm])):
        for j in range(len(output_labels)):
            ax[k].bar(x + (j-1.5)*w, data[:, j], width=w,
                      label=output_labels[j], color=four_colors_set[j])
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(metrics)
        ax[k].set_title(title)
    ax[0].set_ylabel("error (0 = better)")
    ax[1].legend(title="output")
    plt.tight_layout()
    if save_figure:
        plt.savefig(f'figures/{figure_name}.svg', bbox_inches='tight')


def individual_normalized_mse_bar_plot(normalized_metrics_dict,
                                       figure_name, save_figure=False):
    norm = normalized_metrics_dict['MSE']
    x = np.arange(len(norm))
    w = 0.4
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.bar(x, norm, label=output_labels, width=w, color=four_colors_set)
    ax.set_xticks(x)
    ax.set_xticklabels(output_labels)
    ax.set_ylabel("normalized mean squared error")
    plt.tight_layout()
    if save_figure:
        plt.savefig(f'figures/{figure_name}.svg', bbox_inches='tight')


def coefficients_over_iterations_plot(input_coeffs, aft_outputs, nn_outputs):
    """
    Create a line plot showing the evolution of coefficients over iterations.
    Parameters
    ----------
    coefficients_dict : dict
        Dictionary where keys are coefficient names and values are lists of
        coefficient values over iterations.
    figure_name : str
        Name of the figure file to save.
    """
    fig, ax = plt.subplots(4, 1, figsize=(5, 8))
    for i in range(4):
        ax[0].plot(input_coeffs[:, i], color=four_colors_set[i])
        ax[1].plot(aft_outputs[:, i], color=four_colors_set[i])
        ax[2].plot(nn_outputs[:, i], color=four_colors_set[i])
        ax[3].plot(aft_outputs[:, i] - nn_outputs[:, i],
                   label=output_labels[i], color=four_colors_set[i])
    ax[0].set_title('Input coefficients over iterations')
    ax[1].set_title('AFT output coefficients over iterations')
    ax[2].set_title('NN output coefficients over iterations')
    ax[3].legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    ax[3].set_title('Difference of AFT and NN outputs over iterations')
    plt.tight_layout()


def prediction_vs_ground_truth_inset_plot(ground_truth, prediction,
                                          figure_name,
                                          save_figure=False):
    fig, ax = plt.subplots(1, 1, figsize=(4.4, 3))
    fig.subplots_adjust(left=0.15, right=0.7, bottom=0.15, top=0.95)
    for i in range(4):
        for gt, pred in zip(ground_truth, prediction):
            ax.plot(gt[:, i], pred[:, i], '.',
                    label=f'{output_labels[i]}',
                    color=four_colors_set[i])
    ax.set_xlabel('AFT Ground Truth')
    ax.set_ylabel('Neural Network Prediction')
    ax.legend()

    axins = inset_axes(
        ax,
        width=1.2,
        height=1.2,
        loc="lower right",
        bbox_to_anchor=(1.32, 0.15),
        bbox_transform=ax.transAxes,
        borderpad=0
    )
    for i in range(4):
        for gt, pred in zip(ground_truth, prediction):
            axins.plot(gt[:, i], pred[:, i], '.',
                       color=four_colors_set[i])
    axins.set_xlim(0.1, 0.22)
    axins.set_xticks([0.1, 0.2])
    axins.set_ylim(0.1, 0.22)
    axins.set_yticks([0.1, 0.2])
    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")

    if save_figure:
        plt.savefig(f'figures/{figure_name}.svg', bbox_inches='tight')


def frc_plot(frequencies, amplitudes, figure_name, save_figure=False):
    """
    Create a plot of the Frequency Response Function (FRF) with an inset
    zooming in on the region around the resonance peak.
    Parameters
    ----------
    frequencies : np.ndarray
        Array of frequencies to plot.
    amplitudes : np.ndarray
        Array of amplitudes corresponding to the frequencies to plot.
    """
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.2, top=0.9)
    ax.plot(frequencies, amplitudes, linestyle='-', color=two_colors_set[0],
            label='AFT')
    ax.set_xlim(0.4, 1.7)
    ax.set_ylim(0, 3.0)
    ax.set_xlabel(r'Excitation Frequency $\Omega$')
    ax.set_ylabel(r'Amplitude $a$')
    ax.legend(loc='upper left')
    ax.grid()
    if save_figure:
        plt.savefig(f'./figures/{figure_name}.svg', bbox_inches='tight')


def frc_with_inset_plot(frequencies, amplitudes, figure_name,
                        save_figure=False):
    """
    Create a plot of the Frequency Response Function (FRF) with an inset
    zooming in on the region around the resonance peak.
    Parameters
    ----------
    frequencies : list of np.ndarray
        List of frequency arrays to plot.
    amplitudes : list of np.ndarray
        List of amplitude arrays corresponding to the frequencies to plot.
    """
    linestyles = ['-', '']
    markers = ['', '.']
    labels = ['AFT', 'Neural Network']
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.2, top=0.9)
    for freq, amp, ls, marker, color, label in zip(frequencies, amplitudes,
                                                   linestyles, markers,
                                                   two_colors_set, labels):
        ax.plot(freq, amp, linestyle=ls, marker=marker, color=color,
                label=label)
    ax.set_xlim(0.4, 1.7)
    ax.set_ylim(0, 3.0)
    ax.set_xlabel(r'Excitation Frequency $\Omega$')
    ax.set_ylabel(r'Amplitude $a$')
    ax.legend(loc='upper left')
    ax.grid()
    axins = inset_axes(ax,
                       width="30%",
                       height="30%",
                       loc='center left',
                       bbox_to_anchor=(.71, .0, 1.1, 1.2),
                       bbox_transform=ax.transAxes,
                       borderpad=0)
    for freq, amp, ls, marker, color, label in zip(frequencies, amplitudes,
                                                   linestyles, markers,
                                                   two_colors_set, labels):
        axins.plot(freq, amp, linestyle=ls, marker=marker, color=color)
    axins.set_xlim(1.265, 1.272)
    axins.set_ylim(2.8, 2.85)
    axins.set_xticks([1.265, 1.27])
    axins.set_yticks([2.8, 2.83])
    # axins.grid()
    mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5")
    if save_figure:
        plt.savefig(f'./figures/{figure_name}.svg', bbox_inches='tight')


def training_data_distribution_violinplot(inputs, outputs, figure_name,
                                          save_figure=False):
    """
    Create violin plots to visualize the distribution of input and output
    features.
    Parameters
    ----------
    inputs : list of np.ndarray
        List of input feature arrays to plot.
    outputs : list of np.ndarray
        List of output feature arrays to plot.
    """
    x = [1, 2, 5, 6]
    fig, ax = plt.subplots(2, 1, figsize=(5, 6))
    ax[0].violinplot(inputs, positions=x, showmeans=False, showextrema=False)
    ax[0].hlines(0, 0, 6.5, 'k', linewidth=0.5)
    ax[0].set_xlim(0, 6.5)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([r'$a_1$', r'$b_1$', r'$a_3$', r'$b_3$'])
    ax[0].set_title('Distribution of input features')
    ax[1].violinplot(outputs, positions=x, showmeans=False, showextrema=False)
    ax[1].hlines(0, 0, 6.5, 'k', linewidth=0.5)
    ax[1].set_xlim(0, 6.5)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([r'$a_1$', r'$b_1$', r'$a_3$', r'$b_3$'])
    ax[1].set_title('Distribution of output features')
    plt.tight_layout()
    if save_figure:
        plt.savefig(f'./figures/{figure_name}.svg', bbox_inches='tight')


def training_data_distribution_histogram(inputs, outputs, figure_name,
                                         save_figure=False):
    """
    Create histograms to visualize the distribution of input and output
    features.
    Parameters
    ----------
    inputs : list of np.ndarray
        List of input feature arrays to plot.
    outputs : list of np.ndarray
        List of output feature arrays to plot.
    """
    fig, ax = plt.subplots(4, 2, figsize=(10, 6))
    for i in range(np.shape(inputs)[1]):
        ax[i, 0].hist([row[i] for row in inputs], bins=20,
                      color=two_colors_set[0], label=input_labels[i])
        ax[i, 0].legend(loc='upper left')
        ax[i, 1].hist([row[i] for row in outputs], bins=20,
                      color=two_colors_set[1], label=output_labels[i])
        ax[i, 1].legend(loc='upper left')
    ax[1, 0].set_ylabel('Number of samples')
    ax[0, 0].set_title('Distribution of input features')
    ax[1, 1].set_ylabel('Number of samples')
    ax[0, 1].set_title('Distribution of output features')
    plt.tight_layout()
    if save_figure:
        plt.savefig(f'./figures/{figure_name}.svg', bbox_inches='tight')


def frc_and_training_inputs_3d_scatter_plot(frc_inputs,
                                            training_inputs,
                                            figure_name,
                                            save_figure=False):
    """
    Create a 3D scatter plot comparing the distribution of training input
    features with the relevant region for the application.
    Parameters
    ----------
    frc_inputs : np.ndarray
        Array containing the relevant region for the application.
    training_inputs : np.ndarray
        Array containing the training input features.
    figure_name : str
        Name of the figure file to save.
    """
    cmap = LinearSegmentedColormap.from_list("br_custom", four_colors_set)
    fig = plt.figure(figsize=(5, 9))

    frc_indices = [1, 2, 5, 6]  # a1, b1, a3, b3
    a1_frc, b1_frc, a3_frc, b3_frc = frc_inputs[:, frc_indices].T
    train_indices = [0, 1, 2, 3]  # a1, b1, a3, b3
    a1_train, b1_train, a3_train, b3_train = \
        training_inputs[:, train_indices].T

    ax1 = fig.add_subplot(211, projection='3d')
    p1 = ax1.scatter(a1_frc, b1_frc, a3_frc, c=b3_frc, s=8, alpha=0.5,
                     cmap=cmap)
    p1.set_clim(b3_frc.min(), b3_frc.max())
    ax1.set_title("Relevant for application")
    ax1.set_xlabel(r'$a_1$')
    ax1.set_ylabel(r'$b_1$')
    ax1.set_zlabel(r'$a_3$')
    fig.colorbar(p1, ax=ax1, shrink=0.65, label=r'$b_3$', pad=0.15)

    ax2 = fig.add_subplot(212, projection='3d')
    p2 = ax2.scatter(
        a1_train, b1_train, a3_train,
        c=b3_train, s=8, alpha=0.5, cmap=cmap
    )
    p2.set_clim(b3_frc.min(), b3_frc.max())
    ax2.set_title("Training samples")
    ax2.set_xlabel(r'$a_1$')
    ax2.set_ylabel(r'$b_1$')
    ax2.set_zlabel(r'$a_3$')
    fig.colorbar(p2, ax=ax2, shrink=0.65, label=r'$b_3$', pad=0.15)

    a1_all = np.concatenate([a1_frc, a1_train])
    b1_all = np.concatenate([b1_frc, b1_train])
    a3_all = np.concatenate([a3_frc, a3_train])
    ax1.set_xlim(a1_all.min(), a1_all.max())
    ax2.set_xlim(ax1.get_xlim())
    ax1.set_ylim(b1_all.min(), b1_all.max())
    ax2.set_ylim(ax1.get_ylim())
    ax1.set_zlim(a3_all.min(), a3_all.max())
    ax2.set_zlim(ax1.get_zlim())

    plt.tight_layout()
    if save_figure:
        plt.savefig(f'./figures/{figure_name}.png', dpi=300,
                    bbox_inches='tight')


def loss_plot(train_loss, val_loss, figure_name, save_figure=False):
    """
    Create a plot of training and validation loss over epochs.
    Parameters
    ----------
    train_loss : list of float
        List of training loss values over epochs.
    val_loss : list of float
        List of validation loss values over epochs.
    figure_name : str
        Name of the figure file to save.
    """
    plt.figure(figsize=(4, 3))
    plt.semilogy(train_loss, color=two_colors_set[0], label='Training loss')
    plt.semilogy(val_loss, color=two_colors_set[1], label='Validation loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_figure:
        plt.savefig(f'./figures/{figure_name}.svg', bbox_inches='tight')


def aft_process_visualization_plot(k, t, q_F, q_T, q3_T, q3_F, figure_name,
                                   save_figure=False):
    """
    Visualize the AFT process including the original and cubed signals in both
    time and frequency domains.

    Parameters
    ----------
    k : array-like
        Array of frequency indices.
    t : array-like
        Array of time points.
    q_F : array-like
        Fourier coefficients of the original signal.
    q_T : array-like
        Time-domain representation of the original signal.
    q3_T : array-like
        Time-domain representation of the cubed signal.
    q3_F : array-like
        Fourier coefficients of the cubed signal.
    figure_name : str
        Name of the figure file to save.
    """
    fig, ax = plt.subplots(1, 4, figsize=(14, 3))
    ax[0].bar(k, np.real(q_F), 0.4, color='#1D3557', edgecolor='#1D3557')
    ax[0].set_xlabel('k', fontsize=12)
    ax[0].set_title(r'$\widehat{q}(k)$', fontsize=14, fontweight='bold')
    ax[1].plot(np.concatenate([t, t+2*np.pi, t+4*np.pi]), np.tile(q_T, 3), 'k',
               linewidth=5)
    ax[1].set_xlabel('t', fontsize=12)
    ax[1].set_xlim(0, 6*np.pi)
    ax[1].set_title(r'$q(t)$', fontsize=14, fontweight='bold')
    ax[2].plot(np.concatenate([t, t+2*np.pi, t+4*np.pi]), np.tile(q3_T, 3),
               'k', linewidth=5)
    ax[2].set_xlabel('t', fontsize=12)
    ax[2].set_xlim(0, 6*np.pi)
    ax[2].set_title(r'$q^3(t)$', fontsize=14, fontweight='bold')
    ax[3].bar(k, np.real(q3_F), 0.4, color='#1D3557', edgecolor='#1D3557')
    ax[3].set_xlabel('k', fontsize=12)
    ax[3].set_title(r'$\widehat{q^3}(k)$', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_figure:
        plt.savefig(f'./figures/{figure_name}.svg', bbox_inches='tight')


def gradients_over_inputs_plot(inputs, fd_jacobian, nn_jacobian, figure_name,
                               save_figure=False):

    def zero_clean_formatter(x, pos):
        if abs(x) < 1e-12:
            return "0"
        return f"{x:g}"

    idx = [1, 2, 5, 6]  # (a1,b1,a3,b3) in NN-order

    fig, axes = plt.subplots(4, 4, figsize=(10, 7), sharex="col")
    for ii, i in enumerate(idx):  # output index (row)
        for jj, j in enumerate(idx):  # input index (col)
            ax = axes[ii, jj]
            x = inputs[:, j]
            y_fd = fd_jacobian[:, i, j]
            y_nn = nn_jacobian[:, i, j]
            ax.scatter(x, y_fd, s=8, alpha=0.4, color=two_colors_set[0],
                       label="Finite Differences" if (ii == 0 and jj == 0)
                       else None)
            ax.scatter(x, y_nn, s=8, alpha=0.4, color=two_colors_set[1],
                       label="Neural Network" if (ii == 0 and jj == 0)
                       else None)
            if ii == 3:
                ax.set_xlabel(input_labels[jj], fontsize=12)

            ylabel = (fr"$\frac{{\partial {output_symbols[ii]}}}"
                      fr"{{\partial {input_symbols[jj]}}}$")
            ax.set_ylabel(ylabel, rotation=0, fontsize=18, labelpad=15,
                          va="center")

    for ax in axes.flat:
        ax.xaxis.set_major_formatter(FuncFormatter(zero_clean_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(zero_clean_formatter))

    handles, labels_legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.02), fontsize=12, frameon=True)
    fig.subplots_adjust(wspace=0.8, hspace=0.45)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    if save_figure:
        plt.savefig(f"figures/{figure_name}.png", dpi=300, bbox_inches="tight")


def smin_over_omega(smin_list, Omega, min_y, max_y, figure_name,
                    save_figure=False):
    fig, ax = plt.subplots(figsize=(5, 2.2))
    labels = ['Finite Differences', 'Neural Network']
    for smin, color, label in zip(smin_list, two_colors_set, labels):
        ax.semilogy(Omega, smin, '.', label=label, alpha=0.7, color=color)
    ax.set_xlim(0.4, 1.7)
    ax.set_ylim(min_y/10, max_y*10)
    ax.set_xlabel(r"Excitation frequency $\Omega$")
    ax.set_ylabel(r"Smallest singular value $\sigma_{\min}(J)$")
    ax.grid(True)
    fig.legend(loc="lower left", bbox_to_anchor=(0.15, 0.25))
    fig.tight_layout(rect=[0, 0, 1, 1])
    if save_figure:
        plt.savefig(f"figures/{figure_name}.svg", bbox_inches="tight")


def cond_over_omega(cond_list, Omega, min_y, max_y, figure_name,
                    save_figure=False):
    fig, ax = plt.subplots(figsize=(5, 2.2))
    labels = ['Finite Differences', 'Neural Network']
    for cond, color, label in zip(cond_list, two_colors_set, labels):
        ax.semilogy(Omega, cond, '.', label=label, alpha=0.7, color=color)
    ax.set_xlim(0.4, 1.7)
    ax.set_ylim(min_y/10, max_y*10)
    ax.set_xlabel(r"Excitation frequency $\Omega$")
    ax.set_ylabel(r"Condition number $\kappa(J)$")
    ax.grid(True)
    fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.92))
    fig.tight_layout(rect=[0, 0, 1, 1])
    if save_figure:
        plt.savefig(f"figures/{figure_name}.svg", bbox_inches="tight")
