import numpy as np
import numpy.linalg as LA


def Frank_Wolfe(mu, Sigma, rho, x_dim, opts):
    phi_star = {}
    Q_star = {}
    n = x_dim
    iter_max = 1000
    bi_tol = 1e-8
    tol = 1e-4
    verbose = False

    if 'iter_max' in opts:
        iter_max = opts['iter_max']
    if 'bi_tol' in opts:
        bi_tol = opts['bi_tol']
    if 'tol' in opts:
        tol = opts['tol']
    if 'verbose' in opts:
        verbose = opts['verbose']

    def G_(S): return S[:n, n:] @ LA.inv(S[n:, n:]) @ S[n:, :n]
    def f_(S, G): return np.trace(S[:n, :n] - G @ S[n:, :n])

    def grad_f_(G):
        IG = np.concatenate([np.eye(n), -G], axis=1)
        return IG.T @ IG

    S = Sigma
    obj = np.zeros(iter_max)
    res = np.zeros(iter_max)

    if rho == 0:
        G = G_(Sigma)
        phi_star['G'] = G
        phi_star['g'] = mu[:n] - G @ mu[n:]
        Q_star['Sigma'] = Sigma
        Q_star['mu'] = mu
        return phi_star, Q_star, obj, res

    for i in range(iter_max):
        G = G_(S)
        obj_current = f_(S, G)
        D = grad_f_(G)
        L = my_bisection(Sigma, D, rho, bi_tol)

    res_current = abs((L-S).T @ D)
