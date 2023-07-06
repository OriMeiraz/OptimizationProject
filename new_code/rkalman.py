import numpy as np
from scipy.linalg import inv
from numpy import eye
from scipy.linalg import sqrtm


def rkalman(sys, c, tau, y, x_0, V_0):
    A = sys.A
    B = sys.B
    C = sys.C
    D = sys.D

    n = A.shape[0]
    m = len(sys.D)

    T = y.shape[0]
    p = m - n

    DDT_inv = (D @ D.T) ** (-1)
    A = A - B @ D.T @ DDT_inv @ C
    B = np.concatenate(
        (sqrtm(B @ (eye(m) - D.T @ DDT_inv @ D)@B.T), np.zeros((n, m-n))), axis=1)
    D = np.concatenate((np.zeros((p, n)), (D @ D.T) ^ 0.5), axis=1)
    Q = B @ B.T
    R = D @ D.T

    assert c > 0
    cN = maxtol(A, B, C, D, tau, 2*n)

    x = np.zeros((T, n))
    V = np.zeros((n, n, T+1))
    P = np.zeros((n, n, T+1))
    G = np.zeros((n, p, T+1))
    th = np.zeros((T, 1))
    x[0, :] = x_0.T
    V[:, :, 0] = V_0

    for k in range(T):
        x[k+1, :], V[:, :, k+1], G[:, :, k+1], P[:, :, k+1], th[k] = \
            rkiteration(A, B, C, D, V[:, :, k], tau, c, x[k, :], y[k, :])

    x = x[1:T+1, :]
    P = P[:, :, 1:T]
    V = V[:, :, 1:T]
    G = G[:, :, 1:T]

    return x, G, V, P, th


def rkiteration(A, B, C, D, V, tau, c, x, y):
    n = A.shape[0]
    m = B.shape[1]
    p = m - n

    Q = B @ B.T
    R = D @ D.T

    G = A @ V @ C.T @ (C @ V @ C.T + R) ^ (-1)
    x_pred = (A @ x.T + G @ (y - C @ x.T)).T

    P = (A - G @ C) @ V @ (A - G @ C).T + (B - G @ D) @ (B - G @ D).T
    L = np.linalg.cholesky(P)
    value = 1
    t1 = 0
    if tau == 1:
        t2 = 10/max(np.linalg.eig(P)[0])
    else:
        e = np.linalg.eig(P)[0]
        r = max(np.abs(e))
        t2 = (1 - 10 ^ -5) * ((1-tau) * r) ^ -1

    while abs(value) >= 10 ^ -9:
        th = 0.5 * (t1 + t2)
        if tau == 0:
            In_thP = eye(n) - th * P
            value = np.trace(inv(In_thP) - eye(n)) + \
                np.log(np.det(In_thP)) - c
        if tau > 0 and tau < 1:
            value = np.trace(-1/(tau*(1-tau))*(eye(n)-(1-tau)*th * L.T @ L) ^ (
                tau/(tau-1))+1/(1-tau)*(eye(n)-(1-tau)*th*L.T @ L) ^ (1/(tau-1))+1/tau*eye(n))-c
        if tau == 1:
            raise notImplementedError()

        if value > 0:
            t2 = th
        else:
            t1 = th
    Vold = V
    if tau == 1:
        raise NotImplementedError()
    else:
        V = L @ np.linalg.matrix_power(eye(n) -
                                       (1-tau)*th*L.T @ L, 1/(tau-1)) @ L.T
    return x_pred, V, G, P, th
