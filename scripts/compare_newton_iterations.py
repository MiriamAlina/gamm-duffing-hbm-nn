import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Evaluate Newton logs with GLOBALLY increasing newton_iter
# Files:
#   data/newton_log.csv
#   data/newton_log_nn.csv
#
# Columns:
#   Om,newton_iter,normR,stepnorm
# ============================================================

aft_file = "data/newton_log.csv"
nn_file = "data/newton_log_nn.csv"

aft = pd.read_csv(aft_file)
nn = pd.read_csv(nn_file)

aft = aft.replace([np.inf, -np.inf], np.nan).dropna()
nn = nn.replace([np.inf, -np.inf], np.nan).dropna()


# ============================================================
# Reconstruct local Newton iteration from repeated Omega values
# (works if each continuation step logs several Newton corrections
#  at the same/similar Omega before moving on)
# ============================================================
def reconstruct_local_newton(df, om_tol=1e-12):
    df = df.copy().reset_index(drop=True)

    step_id = np.zeros(len(df), dtype=int)
    local_iter = np.zeros(len(df), dtype=int)

    current_step = 0
    current_iter = 1
    step_id[0] = current_step
    local_iter[0] = current_iter

    for i in range(1, len(df)):
        same_om = abs(df.loc[i, "Om"] - df.loc[i - 1, "Om"]) < om_tol

        if same_om:
            current_iter += 1
        else:
            current_step += 1
            current_iter = 1

        step_id[i] = current_step
        local_iter[i] = current_iter

    df["step_id"] = step_id
    df["local_newton_iter"] = local_iter
    return df


aft = reconstruct_local_newton(aft)
nn = reconstruct_local_newton(nn)


# ============================================================
# Summaries
# ============================================================
def summarize(df, name):
    step_stats = df.groupby("step_id").agg(
        Om=("Om", "first"),
        n_newton=("local_newton_iter", "max"),
        final_normR=("normR", "last"),
        final_stepnorm=("stepnorm", "last"),
    )

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Logged Newton evaluations:      {len(df)}")
    print(f"Continuation steps detected:    {len(step_stats)}")
    print("Mean Newton iterations/step:    " +
          f"{step_stats['n_newton'].mean():.3f}")
    print("Median Newton iterations/step:  " +
          f"{step_stats['n_newton'].median():.3f}")
    print("Max Newton iterations/step:     " +
          f"{step_stats['n_newton'].max()}")
    print("Mean final step norm:           " +
          f"{step_stats['final_stepnorm'].mean():.3e}")
    print("Mean final residual norm:       " +
          f"{step_stats['final_normR'].mean():.3e}")

    return step_stats


aft_steps = summarize(aft, "AFT")
nn_steps = summarize(nn, "NN")


# ============================================================
# 1) Newton iterations per continuation step vs Omega
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(aft_steps["Om"], aft_steps["n_newton"], ".", label="AFT",
             alpha=0.7)
axes[0].plot(nn_steps["Om"], nn_steps["n_newton"], ".", label="NN", alpha=0.7)
axes[0].set_xlabel(r"$\Omega$")
axes[0].set_ylabel("Newton iterations per step")
axes[0].set_title("Newton iterations vs Ω")
axes[0].grid(True)

axes[1].semilogy(aft_steps["Om"], aft_steps["final_stepnorm"], ".",
                 label="AFT",
                 alpha=0.7)
axes[1].semilogy(nn_steps["Om"], nn_steps["final_stepnorm"], ".", label="NN",
                 alpha=0.7)
axes[1].set_xlabel(r"$\Omega$")
axes[1].set_ylabel(r"final $\|\Delta X\|$")
axes[1].set_title("Final Newton step norm vs Ω")
axes[1].grid(True)

axes[2].semilogy(aft_steps["Om"], aft_steps["final_normR"], ".", label="AFT",
                 alpha=0.7)
axes[2].semilogy(nn_steps["Om"], nn_steps["final_normR"], ".", label="NN",
                 alpha=0.7)
axes[2].set_xlabel(r"$\Omega$")
axes[2].set_ylabel(r"final $\|R\|$")
axes[2].set_title("Final residual norm vs Ω")
axes[2].grid(True)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2)
fig.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("figures/newton_per_step_vs_omega.png", dpi=300,
            bbox_inches="tight")
plt.show()


# ============================================================
# 2) Histograms of Newton iterations per continuation step
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

bins = np.arange(
    0.5,
    max(aft_steps["n_newton"].max(), nn_steps["n_newton"].max()) + 1.5,
    1
)

axes[0].hist(aft_steps["n_newton"], bins=bins, alpha=0.6, label="AFT")
axes[0].hist(nn_steps["n_newton"], bins=bins, alpha=0.6, label="NN")
axes[0].set_xlabel("Newton iterations per step")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of Newton iterations")
axes[0].grid(True)

axes[1].hist(aft_steps["final_stepnorm"], bins=40, alpha=0.6, label="AFT")
axes[1].hist(nn_steps["final_stepnorm"], bins=40, alpha=0.6, label="NN")
axes[1].set_xscale("log")
axes[1].set_xlabel(r"final $\|\Delta X\|$")
axes[1].set_ylabel("Count")
axes[1].set_title("Distribution of final step norms")
axes[1].grid(True)

axes[2].hist(aft_steps["final_normR"], bins=40, alpha=0.6, label="AFT")
axes[2].hist(nn_steps["final_normR"], bins=40, alpha=0.6, label="NN")
axes[2].set_xscale("log")
axes[2].set_xlabel(r"final $\|R\|$")
axes[2].set_ylabel("Count")
axes[2].set_title("Distribution of final residuals")
axes[2].grid(True)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2)
fig.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("figures/newton_per_step_histograms.png", dpi=300,
            bbox_inches="tight")
plt.show()


# ============================================================
# 3) Residual convergence within Newton corrections
# ============================================================
def plot_convergence(df, title, filename, max_steps=25):
    fig, ax = plt.subplots(figsize=(6, 4))

    grouped = df.groupby("step_id")
    shown = 0
    for _, g in grouped:
        if shown >= max_steps:
            break
        g = g.sort_values("local_newton_iter")
        ax.semilogy(g["local_newton_iter"], g["normR"], "-o", ms=3, alpha=0.5)
        shown += 1

    ax.set_xlabel("Local Newton iteration")
    ax.set_ylabel(r"$\|R\|$")
    ax.set_title(title)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


plot_convergence(aft, "Residual convergence within Newton steps (AFT)",
                 "figures/newton_convergence_aft.png")

plot_convergence(nn, "Residual convergence within Newton steps (NN)",
                 "figures/newton_convergence_nn.png")
