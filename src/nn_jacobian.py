import numpy as np
import torch
from src.fourier_conversion import convert_comexp_to_cossin


def NN_jacobian(input, mu, zeta, kappa, gamma, P, H, N, nonlinear_term, NN_id,
                evaluate_coefficients, Om):

    omega_h = np.arange(-H, H+1) * Om  # Harmonic frequencies

    NN_model = torch.load('models/MLP_Duffing_'+NN_id+'.pt',
                          weights_only=False)
    NN_model.eval()

    input_tensor = torch.tensor(input, dtype=torch.float32, requires_grad=True)
    jac = torch.autograd.functional.jacobian(NN_model, input_tensor)
    linear_term_ce = -mu * omega_h**2 + 1j * omega_h * zeta + kappa
    # Conversion of complex-exponential to sine-cosine representation
    linear_term_cs = convert_comexp_to_cossin(linear_term_ce, H)
    linear_term = np.diag(linear_term_cs)
    derivative = linear_term + jac.detach().numpy() * 0.001

    return derivative


def NN_jacobian_Duffing_H3(input, mu, zeta, kappa, gamma, P, H, N,
                           nonlinear_term, NN_id, evaluate_coefficients, Om):
    H = int(H)

    S = np.zeros((2*H+1, 2*H+1))
    S[0, 0] = kappa
    for n in range(1, H+1):
        Kn = kappa - mu * (n * Om)**2
        Cn = zeta * n * Om
        i = 1 + 2*(n-1)
        S[i, i] = Kn
        S[i, i+1] = Cn
        S[i+1, i] = -Cn
        S[i+1, i+1] = Kn

    NN_model = torch.load('models/MLP_Duffing_H3_'+NN_id+'.pt',
                          weights_only=False)
    NN_model.eval()
    relevant_input = np.concatenate((input[1:3], input[5:7]))
    input_tensor = torch.tensor(relevant_input, dtype=torch.float32,
                                requires_grad=True)
    jac = torch.autograd.functional.jacobian(NN_model, input_tensor)
    dFnl = np.zeros((2*H+1, 2*H+1))
    dFnl[1:3, 1:3] = jac[0:2, 0:2]
    dFnl[5:, 1:3] = jac[2:, 0:2]
    dFnl[1:3, 5:] = jac[0:2, 2:]
    dFnl[5:, 5:] = jac[2:, 2:]
    derivative = S + dFnl

    return derivative
