import numpy as np
from src.plotting import aft_process_visualization_plot


H = 3
N = 256

k = np.linspace(1, H, H)
t = np.linspace(0, 2*np.pi, N, endpoint=False)

q_F = [0, 20, 2]

Q = np.zeros(N, dtype=complex)
for n, v in enumerate(q_F):
    Q[n % N] = v
q_T = np.fft.ifft(Q).real

q3_T = q_T**3

q3_F = np.fft.fft(q3_T)

aft_process_visualization_plot(k, t, q_F, q_T, q3_T, q3_F[:H],
                               figure_name='aft_process_visualization',
                               save_figure=False)
