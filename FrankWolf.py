import numpy as np
from numpy import linalg as LA
from utils import calc_L, matrix_dot, get_LB_UB, get_delta


def bisection(cov, D, radius, tol):
    d = D.shape[0]

    def h(gamma):
        temp = gamma*np.eye(d) - D
        temp = LA.inv(temp)
        temp = np.eye(d) - gamma*temp
        temp = temp @ temp
        temp = matrix_dot(temp, cov)
        return radius**2 - temp
    LB, UB = get_LB_UB(cov, D, radius)

    while True:
        gamma = (LB + UB)/2
        L = calc_L(gamma, D, cov)
        h_gamma = h(gamma)
        if h_gamma < 0:
            LB = gamma
        else:
            UB = gamma
        delta = get_delta(gamma, D, cov, radius, L)
        if h_gamma > 0 and delta < tol:
            break

    return L
