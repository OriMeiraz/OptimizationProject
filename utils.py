import numpy as np
from numpy import linalg as LA
import argparse
from scipy.linalg import inv


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    print(v)
    raise argparse.ArgumentTypeError(f'Boolean value expected - got {v}.')


def calc_L(gamma, D, cov, Id=None):
    if Id is None:
        Id = np.eye(D.shape[0])
    diff = gamma*Id - D
    diff_inv = inv(diff)
    return gamma * gamma * (diff_inv @ cov @ diff_inv), diff_inv


def matrix_dot(A, B):
    return (A * B).sum()


def get_LB_UB(cov, D, radius):
    eigvals, eigvecs = LA.eigh(D)
    lambda_1 = eigvals[-1]
    v_1 = eigvecs[:, -1]
    LB = lambda_1 * (1+np.sqrt(v_1.T @ cov @ v_1)/radius)
    UB = lambda_1 * (1 + np.sqrt(np.trace(cov))/radius)
    return LB, UB


def get_delta(gamma, D, cov, radius, L, diff_inv=None):
    if diff_inv is None:
        diff_inv = LA.inv(gamma*np.eye(D.shape[0]) - D)
    d = D.shape[0]
    return gamma * (radius**2 - cov.trace()) - matrix_dot(L, D) +\
        gamma**2 * matrix_dot(diff_inv, cov)


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
    assert np.isclose(sigma, sigma.T).all()
    sigma = (sigma + sigma.T)/2
    return mu, sigma
