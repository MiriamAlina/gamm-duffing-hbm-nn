import numpy as np
from scipy.spatial.distance import cosine


def cosine_similarity(x1, x2):
    # Definition:
    # (np.sum(x1 * x2)) / (np.sqrt(np.sum(x1 ** 2)) * np.sqrt(np.sum(x2 ** 2)))
    result = 1 - cosine(x1, x2)

    return result


def cosine_similarity_complex(x1, x2):

    result = np.real(np.vdot(x1, x2)) / (np.linalg.norm(x1) *
                                         np.linalg.norm(x2))

    return result
