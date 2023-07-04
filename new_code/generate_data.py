import numpy as np


def generate_data(sys, x_0, T, coeff, is_TV):
    n = sys.A.shape[0]
    if len(sys.D.shape) == 1:
        m = 1
    else:
        m = sys.D.shape[1]
    d = m + n
    r0 = np.random.randn(d)
    y0 = sys.C @ x_0 + sys.D @ r0
    x_prev = x_0
    x = np.zeros((n, T))
    y = np.zeros((m, T))
    Delta = 2 * np.random.rand() - 1
    A_purt = sys.A + np.array([[0, coeff * Delta], [0, 0]])
    for t in range(T):
        x[:, t] = A_purt @ x_prev + sys.B @ np.random.randn(d)
        y[:, t] = sys.C @ x[:, t] + sys.D @ np.random.randn(d)
        x_prev = x[:, t]
        if is_TV:
            Delta = 2 * np.random.rand() - 1
            A_purt = sys.A + np.array([[0, coeff * Delta], [0, 0]])
    return x, y, y0
