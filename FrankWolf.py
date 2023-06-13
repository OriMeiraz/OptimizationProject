import numpy as np
from numpy import linalg as LA


def __calc_L(gamma, D, cov):
    diff = gamma - D
    diff_inv = LA.inv(diff)
    return gamma**2 * (diff_inv @ cov @ diff_inv)


def bisection(cov, D, radius, tol):
    def h(gamma):
        raise NotImplementedError("TODO: finish this function")
        pass

    eigvals, eigvecs = LA.eigh(D)
    delta_1 = eigvals[-1]
    v_1 = eigvecs[:, -1]
    LB = delta_1 * (1+np.sqrt(v_1.T @ cov @ v_1)/radius)
    UB = delta_1 * (1 + LA.trace(cov)/radius)

    while True:
        gamma = (LB + UB)/2
        raise NotImplementedError("TODO: finish this function")
