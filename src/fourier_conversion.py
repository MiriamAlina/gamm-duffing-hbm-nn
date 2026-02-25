import numpy as np


def convert_comexp_to_cossin(ce, H):
    cs = np.concatenate((
        [np.real(ce[H])],
        np.ravel(np.column_stack((2 * np.real(ce[H+1:]),
                                  -2 * np.imag(ce[H+1:]))))))
    return cs


def convert_cossin_to_comexp(cs):

    cos_part = cs[1:-1:2]  # Cosine terms of harmonics
    sin_part = cs[2::2]  # Sine terms of harmonics
    ce = np.concatenate((np.flipud(cos_part + 1j * sin_part) / 2,
                         np.array([cs[0]]),
                         (cos_part - 1j * sin_part) / 2))
    return ce
