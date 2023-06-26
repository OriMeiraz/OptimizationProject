import numpy as np
import exp1_utils as eu
import utils
import Algorithms as alg
from tqdm import trange, tqdm


def run(time_var: bool, small: bool, radius, total: int = 1000):
    get_uncertainty = eu.getUncertainty(time_var, small, total)
    C = np.array([1, -1])
    DDT = np.array([1])
    BDT = np.array([0, 0])
    BBT = np.array([[1.9608, 0.0195],
                    [0.0195, 1.9605]])
    V = np.eye(2)
    x_hat = eu.get_x0_hat()
    for _ in trange(total):
        A = utils.get_At(get_uncertainty)
        V, x_hat = alg.robustKalmanFilter(
            V, x_hat, radius, tol, A, BBT, C, DDT, BDT, n)

        raise NotImplementedError('finish run')


if __name__ == '__main__':
    run(False, False, 1000)
