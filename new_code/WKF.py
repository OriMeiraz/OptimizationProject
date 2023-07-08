import numpy as np
from Frank_Wolfe import Frank_Wolfe
from tqdm import trange


def WKF(sys, rho, Y_t, x_0, V_0, opts=None):
    if opts is None:
        opts = {}
    n = sys.A.shape[0]
    if len(sys.D.shape) == 1:
        m = 1
    else:
        m = sys.D.shape[1]
    T = Y_t.shape[1]

    d = n + m

    x_prev = x_0
    V_prev = V_0
    xhat = np.zeros((n, T))
    V = np.zeros((n, n, T))
    G = np.zeros((n, m, T))
    S = np.zeros((d, d, T))

    for t in trange(T, leave=False, desc='WKF'):
        mu_t, Sigma_t = predict(
            x_prev, V_prev, sys.A, sys.B, sys.C, sys.D)

        if isinstance(rho, np.ndarray):
            rho_t = rho[t]
        else:
            rho_t = rho

        xhat[:, t], V[:, :, t], G[:, :, t], S[:, :, t] = update(
            mu_t, Sigma_t, rho_t, Y_t[:, t], len(x_0), opts)

        x_prev = xhat[:, t]
        V_prev = V[:, :, t]

    return xhat, V, G, S


def predict(x_prev, V_prev, A, B, C, D):
    A_aug = np.vstack([A, C @ A])
    B_aug = np.vstack([B, C @ B + D])

    mu_t = A_aug @ x_prev
    Sigma_t = A_aug @ V_prev @ A_aug.T + B_aug @ B_aug.T
    return mu_t, Sigma_t


def update(mu_t, Sigma_t, rho_t, y_t, n, opts):
    phi_star, Q_star, _, _ = Frank_Wolfe(mu_t, Sigma_t, rho_t, n, opts)
    G_t = phi_star['G']
    S_t = Q_star['Sigma']
    V_t = S_t[:n, :n] - G_t @ S_t[n:, :n]
    xhat_t = G_t @ (y_t - mu_t[n:]) + mu_t[:n]
    return xhat_t, V_t, G_t, S_t
