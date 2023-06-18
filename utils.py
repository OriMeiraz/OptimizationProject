import numpy as np
from numpy import linalg as LA


def calc_L(gamma, D, cov):
    diff = gamma*np.eye(D.shape[0]) - D
    diff_inv = LA.inv(diff)
    return gamma**2 * (diff_inv @ cov @ diff_inv)


def matrix_dot(A, B):
    return np.trace(A @ B)


def get_LB_UB(cov, D, radius):
    eigvals, eigvecs = LA.eigh(D)
    delta_1 = eigvals[-1]
    v_1 = eigvecs[:, -1]
    LB = delta_1 * (1+np.sqrt(v_1.T @ cov @ v_1)/radius)
    UB = delta_1 * (1 + LA.trace(cov)/radius)
    return LB, UB


def get_delta(gamma, D, cov, radius, L):
    d = D.shape[0]
    return gamma * (radius**2 - cov.trace()) - matrix_dot(L, D) +\
        gamma**2 * matrix_dot(LA.inv(gamma * np.eye(d) - D), cov)
