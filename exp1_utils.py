import numpy as np


class getUncertainty():
    def __init__(self, time_var: bool, small: bool, total: int = 1000):
        self.time_var = time_var
        self.size = 1 if small else 10
        self.t = 0
        if not time_var:
            self.d = [np.random.uniform(-self.size, self.size)] * total
        else:
            self.d = np.random.uniform(-self.size, self.size, total)

    def next(self):
        ret = self.d[self.t]
        self.t += 1
        return ret


def get_At(get_uncertainty):
    d = get_uncertainty.next()
    A = np.array([[0.9802, 0.0196+0.099*d],
                  [0, 0.9802]])
    return A


def get_x0_hat():
    return np.random.multivariate_normal(np.zeros(2), np.eye(2))


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
    return mu, sigma
