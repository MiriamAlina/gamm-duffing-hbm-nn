import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 4, figsize=(14, 3))
x = np.linspace(0, 20, 1000)

H = 3
k = np.linspace(1, H, H)
q_F = [0, 20, 2]
ax[0].bar(k, np.real(q_F), 0.4, color='#1D3557', edgecolor='#1D3557')
ax[0].set_xlabel('k', fontsize=12)
ax[0].set_title(r'$\widehat{q}(k)$', fontsize=14, fontweight='bold')

N = 256
Q = np.zeros(N, dtype=complex)
for n, v in enumerate(q_F):
    Q[n % N] = v
q_T = np.fft.ifft(Q).real
t = np.linspace(0, 2*np.pi, N, endpoint=False)
ax[1].plot(np.concatenate([t, t+2*np.pi, t+4*np.pi]), np.tile(q_T, 3), 'k')
ax[1].set_xlabel('t', fontsize=12)
ax[1].set_xlim(0, 6*np.pi)
ax[1].set_title(r'$q(t)$', fontsize=14, fontweight='bold')

q3_T = q_T**3
ax[2].plot(np.concatenate([t, t+2*np.pi, t+4*np.pi]), np.tile(q3_T, 3), 'k')
ax[2].set_xlabel('t', fontsize=12)
ax[2].set_xlim(0, 6*np.pi)
ax[2].set_title(r'$q^3(t)$', fontsize=14, fontweight='bold')

q3_F = np.fft.fft(q3_T)
ax[3].bar(k, np.real(q3_F[:H]), 0.4, color='#1D3557', edgecolor='#1D3557')
ax[3].set_xlabel('k', fontsize=12)
ax[3].set_title(r'$\widehat{q^3}(k)$', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/aft_visualization.svg', bbox_inches='tight')
plt.show()
