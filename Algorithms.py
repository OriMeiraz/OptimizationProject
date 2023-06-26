import numpy as np
from numpy import linalg as LA
from utils import calc_L, matrix_dot, get_LB_UB, get_delta
from math import sqrt


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


def frank_wolfe(cov, radius, tol, n: int, max_iter=1000):
    sigma_low = np.min(LA.eigvals(cov))
    sigma_high = (radius + sqrt(cov.trace()))**2
    C_high = 2 * np.power(sigma_high, 4) / np.power(sigma_low, 3)
    S = cov
    k = 0
    stoping_criterion = False
    while k < max_iter and not stoping_criterion:
        alpha = 2/(k+2)
        S_xy = S[:n, n:]
        S_yy = S[n:, n:]
        G = S_xy @ LA.inv(S_yy)
        In_G = np.concatenate((np.eye(n), G), axis=1)
        D = In_G.T @ In_G
        eps = alpha * tol * C_high
        L = bisection(cov, D, radius, eps)
        S += alpha * (L - S)
        k += 1
        stoping_criterion = True
        raise NotImplementedError()
    return S


def robustKalmanFilter(cov, state, radius, tol, A, BB, C, DD, BD, n: int, max_iter=1000):
    pass