import numpy as np
from scipy.linalg import inv, solve_discrete_are as dare
from numpy import eye
from scipy.linalg import sqrtm
from numpy import linalg as LA
from tqdm import trange


def maxtol(A, B, C, D, tau, N):
    n = A.shape[0]
    m = B.shape[1]
    p = m - n
    1 == 1
    DR = np.kron(eye(N), D)
    Re = None
    Ob = None
    ObR = None
    H = None
    L = None

    for k in range(N):
        Ak = LA.matrix_power(A, k)
        Re = np.hstack([Re,  Ak @ B]) if Re is not None else Ak @ B
        Ob = np.vstack([C @ Ak, Ob]) if Ob is not None else C @ Ak
        ObR = np.vstack([Ak, ObR]) if ObR is not None else Ak

        T = None

        for l in range(1, N+1):
            if l <= N-k:
                T = np.hstack([T, np.zeros((p, m))]
                              ) if T is not None else np.zeros((p, m))
            else:
                Al = LA.matrix_power(A, l-N + k - 1)
                T = np.hstack([T, [C @ Al @ B]]
                              ) if T is not None else C @ Al @ B

        H = np.vstack([T, H]) if H is not None else T
        T = None
        for l in range(1, N+1):
            if l <= N-k:
                T = np.hstack([T, np.zeros((n, m))]
                              ) if T is not None else np.zeros((n, m))
            else:
                Al = LA.matrix_power(A, l-N + k - 1)
                T = np.hstack([T, Al @ B]) if T is not None else Al @ B
        L = np.vstack([T, L]) if L is not None else T
    HD_inv = inv(DR @ DR.T + H @ H.T)
    Om = Ob.T @ HD_inv @ Ob
    J = ObR - L @ H.T @ HD_inv @ Ob
    M = L @ inv(eye(N * m) + H.T @ inv(DR @ DR.T) @ H) @ L.T

    phiNtilde = 1/max(LA.eigvals(M))

    value = 1
    t1 = 0
    t2 = (1-1e-10) * phiNtilde

    while abs(value) >= 1e-9:
        theta = 0.5 * (t1 + t2)
        Sth = - eye(N * n) + theta * M
        Omth = Om + J.T * theta @ inv(Sth) @ J
        value = -min(LA.eigvals(Omth))
        if value > 0:
            t2 = theta
        else:
            t1 = theta

    phiN = min(phiNtilde, theta)

    Pq = dare(A.T, C.reshape(-1, 1), B @ B.T, D @ D.T)
    eigs = LA.eigvals(Pq)
    lamb = min(eigs)

    if tau == 1:
        thN = -np.log(1-phiN*lamb)/lamb
    else:
        thN = (1-(1-lamb * phiN)**(1-tau))/((1-tau)*lamb)

    if tau >= 0 and tau < 1:
        thNmax = ((1-tau)*max(eigs)) ** -1
    else:
        thNmax += np.inf

    if thN > thNmax:
        cN += np.inf

    else:
        if tau == 0:
            cN = (np.log(LA.det(eye(n)-thN*Pq))+np.trace(inv(eye(n)-thN*Pq))-n)

        if tau > 0 and tau < 1:
            raise NotImplementedError()

        if tau == 1:
            raise NotImplementedError()

    return cN


def rkalman(sys, c, tau, y, x_0, V_0):
    A = sys.A
    B = sys.B
    C = sys.C
    D = sys.D

    n = A.shape[0]
    m = len(sys.D)

    T = y.shape[0]
    p = m - n

    DDT_inv = 1/(D @ D.T)
    A = A - B @ D.T * DDT_inv @ C
    B = np.concatenate(
        (sqrtm(B @ (eye(m) - D.reshape(-1, 1) * DDT_inv @ D.reshape(1, -1))@B.T), np.zeros((n, m-n))), axis=1)
    D = np.hstack((np.zeros((p, n)), np.ones((p, 1)) * (D @ D.T) ** 0.5))
    Q = B @ B.T
    R = D @ D.T

    assert c > 0
    cN = maxtol(A, B, C, D, tau, 2*n)

    x = np.zeros((T+1, n))
    V = np.zeros((n, n, T+1))
    P = np.zeros((n, n, T+1))
    G = np.zeros((n, p, T+1))
    th = np.zeros((T, 1))
    x[0, :] = x_0.T
    V[:, :, 0] = V_0
    y = y.reshape(-1, 1)

    for k in trange(T, desc='KL iteration', leave=False):
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

    G = A @ V @ C.T * (C @ V @ C.T + R) ** (-1)
    x_pred = (A @ x.T + G * (y - C @ x.T)).T
    GD = np.kron(G, D).reshape(2, -1)
    GC = np.outer(G, C)
    P = (A - GC) @ V @ (A - GC).T + (B - GD) @ (B - GD).T
    L = np.linalg.cholesky(P)
    value = 1
    t1 = 0
    if tau == 1:
        t2 = 10/max(np.linalg.eig(P)[0])
    else:
        e = np.linalg.eig(P)[0]
        r = max(np.abs(e))
        t2 = (1 - 10 ** -5) * ((1-tau) * r) ** -1

    while abs(value) >= 10 ** -9:
        th = 0.5 * (t1 + t2)
        if tau == 0:
            In_thP = eye(n) - th * P
            value = np.trace(inv(In_thP) - eye(n)) + \
                np.log(LA.det(In_thP)) - c
        if tau > 0 and tau < 1:
            value = np.trace(-1/(tau*(1-tau))*(eye(n)-(1-tau)*th * L.T @ L) ** (
                tau/(tau-1))+1/(1-tau)*(eye(n)-(1-tau)*th*L.T @ L) ** (1/(tau-1))+1/tau*eye(n))-c
        if tau == 1:
            raise notImplementedError()

        if value > 0:
            t2 = th
        else:
            t1 = th
    Vold = V
    if tau == 1:
        raise NotImplementedError()
    if 0 < tau < 1:
        raise NotImplementedError()
    if tau == 0:
        V = L @ inv(eye(n) - (1-tau)*th*L.T @ L) @ L.T
    return x_pred.T, V, G.T, P, th
