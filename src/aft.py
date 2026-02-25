import numpy as np


def compute_AFT_solution(N, H, Q_ce, gamma):
    """
    Compute Fourier coefficients of nonlinear forcing from displacement
    coefficients with a given nonlinearity.
    Inputs:
        N: Number of time samples (int)
        H: Number of harmonics (int)
        Q_ce: Fourier coefficients of displacement (complex array)
        gamma: Nonlinearity coefficient (float)
    Returns:
        Fnl_ce: Fourier coefficients of nonlinear force (complex array)
    """

    # Inverse DFT matrix (time domain reconstruction)
    n = np.arange(N).reshape(-1, 1)  # Column vector for time samples
    h = np.arange(-H, H+1).reshape(1, -1)  # Row vector for harmonics
    E_NH = np.exp(1j * 2 * np.pi / N * n @ h)

    # Time-domain response q(t)
    q = np.real(E_NH @ Q_ce)

    # Nonlinear force in time domain (Duffing nonlinearity)
    fnl = gamma * q**3

    # Fourier coefficients of nonlinear force (DFT)
    Fnl_ce = (E_NH.conj().T @ fnl) / N

    return Fnl_ce
