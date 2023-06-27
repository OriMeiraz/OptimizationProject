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


def get_x0():
    return np.random.multivariate_normal(np.zeros(2), np.eye(2))


def get_x0_hat():
    return np.array([0, 0])
