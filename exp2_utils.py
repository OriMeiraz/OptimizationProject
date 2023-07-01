import numpy as np
from scipy.signal import convolve


def smooth(a, WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid')/WSZ
    r = np.arange(1, WSZ-1, 2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((start, out0, stop))


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
