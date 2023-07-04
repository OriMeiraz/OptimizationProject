import numpy as np
import numpy.linalg as LA
from scipy.linalg import inv

def Frank_Wolfe(mu, Sigma, rho, x_dim, opts=None):
    if opts is None:
        opts = {}
    phi_star = {}
    Q_star = {}
    n = x_dim
    iter_max = 1000
    bi_tol = 1e-8
    tol = 1e-4

    if 'iter_max' in opts:
        iter_max = opts['iter_max']
    if 'bi_tol' in opts:
        bi_tol = opts['bi_tol']
    if 'tol' in opts:
        tol = opts['tol']

    def G_(S):
        S_xy = S[n:, n:]
        if S_xy.size == 1:
            return S[:n, n:] / S_xy
        return S[:n, n:] @ LA.inv(S_xy)

    def f_(S, G):
        return np.trace(S[:n, :n] - G @ S[n:, :n])

    def vec(x):
        return x.reshape(-1)

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

        res_current = abs(vec(L-S) @ vec(D))
        if res_current / obj_current < tol:
            break

        alpha = 2 / (i + 2)
        S = alpha * L + (1 - alpha) * S
        if i > 0:
            obj[i] = obj_current
            res[i] = abs(vec(L-S) @ vec(D))

        phi_star['G'] = G
        phi_star['g'] = mu[:n] - G @ mu[n:]
        Q_star['Sigma'] = S
        Q_star['mu'] = mu
    if i < iter_max:
        obj = obj[:i]
        res = res[:i]
    phi_star['G'] = G
    phi_star['g'] = mu[:n]-G@mu[n:]
    Q_star['Sigma'] = S
    Q_star['mu'] = mu
    return phi_star, Q_star, obj, res


def my_bisection(Sigma, D, rho, bi_tol):
    d = D.shape[0]
    Id = np.eye(d)

    def h(inv_D):
        IminusD = Id - inv_D
        return rho*rho - vec(Sigma) @ vec(IminusD @ IminusD)

    def vec(x): return x.reshape(-1)

    def calc_L(D_inv, Sigma):
        return LA.multi_dot([D_inv, Sigma, D_inv])

    values, vectors = LA.eigh(D)
    v_1 = vectors[:, -1]
    lambda_1 = values[-1]

    LB = lambda_1 * (1 + np.sqrt(v_1 @ Sigma @ v_1)/rho)
    UB = lambda_1 * (1 + np.sqrt(np.trace(Sigma))/rho)

    while True:
        gamma = (LB + UB) / 2
        D_inv = gamma * inv(gamma * Id - D)
        L = calc_L(D_inv, Sigma)
        h_val = h(D_inv)

        if h_val < 0:
            LB = gamma
        else:
            UB = gamma

        Delta = gamma * (rho*rho - Sigma.trace()) + gamma * \
            (D_inv * Sigma).sum() - (L * D).sum()

        if h_val >= 0 and Delta < bi_tol:
            break

    return L
