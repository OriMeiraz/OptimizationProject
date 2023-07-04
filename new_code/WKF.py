import numpy as np
import numpy.linalg as LA
from Frank_Wolfe import Frank_Wolfe


def WKF(sys, rho, Y_t, x_0, V_0, opts):
    n = sys['A'].shape[0]
    m = sys['B'].shape[1]
    T = len(Y_t)

    d = n + m

    x_prev = x_0
    V_prev = V_0
    xhat = np.zeros((n, T))
    V = np.zeros((n, n, T))
    G = np.zeros((n, m, T))
    S = np.zeros((d, d, T))

    for t in range(T):
        mu_t, Sigma_t = predict(
            x_prev, V_prev, sys['A'], sys['B'], sys['C'], sys['D'])

        try:
            rho_t = rho[t]
        except IndexError:
            rho_t = rho

        xhat[:, t], V[:, :, t], G[:, :, t], S[:, :, t] = update(
            mu_t, Sigma_t, rho_t, Y_t[:, t], len(x_0), opts)

        x_prev = xhat[:, t]
        V_prev = V[:, :, t]

    return xhat, V, G, S


def predict(x_prev, V_prev, A, B, C, D):
    A_aug = np.concatenate([A, C @ A], axis=0)
    B_aug = np.concatenate([B, C @ B + D], axis=0)
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
