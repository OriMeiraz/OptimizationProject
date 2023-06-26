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


def calc_right_sigma(BBT, C, DDT, BDT):
    if tuple(C.shape) == (2, 1):
        C = C.T
    elif len(C.shape) == 1:
        C = C.reshape(1, -1)

    if tuple(BDT.shape) == (1, 2):
        BDT = BDT.T
    elif len(BDT.shape) == 1:
        BDT = BDT.reshape(-1, 1)

    first = BBT
    second = BBT @ C.T + BDT
    third = second.T
    temp = C @ BDT
    fourth = C @ BBT @ C.T +\
        temp + temp.T + DDT
    return np.block([[first, second], [third, fourth]])


def calc_left_sigma(A, C, V):
    if tuple(C.shape) == (2, 1):
        C = C.T
    elif len(C.shape) == 1:
        C = C.reshape(1, -1)

    top = A
    bottom = C @ A
    block = np.block([[top], [bottom]])
    return block @ V @ block.T


def calc_sigma(A, BBT, C, DDT, BDT, V):
    left = calc_left_sigma(A, C, V)
    right = calc_right_sigma(BBT, C, DDT, BDT)
    return left + right


def calc_mu(A, C, x):
    if tuple(C.shape) == (2, 1):
        C = C.T
    elif len(C.shape) == 1:
        C = C.reshape(1, -1)
        top = A

    bottom = C @ A
    block = np.block([[top], [bottom]])
    return block @ x


def get_mu_sigma(A, BBT, C, DDT, BDT, V, x):
    mu = calc_mu(A, C, x)
    sigma = calc_sigma(A, BBT, C, DDT, BDT, V)
    return mu, sigma
