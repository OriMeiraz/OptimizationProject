import numpy as np


def matrix_inner_product(A, B):
    """
    spectral inner product
    :param A:
    :param B:
    :return: Trace[A^T B]
    """
    return np.trace(np.matmul(A, B))
