import numpy as np
import exp1_utils as eu
import Algorithms as alg
from tqdm import trange
import pandas as pd
import utils
from scipy.linalg import sqrtm


def run(time_var: bool, small: bool, radius, tol, total: int = 1000):
    all_v_x = []
    get_uncertainty = eu.getUncertainty(time_var, small, total)
    C = np.array([1, -1])
    DDT = np.array([1])
    BDT = np.array([0, 0])
    BBT = np.array([[1.9608, 0.0195],
                    [0.0195, 1.9605]])
    B = np.concatenate((sqrtm(BBT), np.zeros((2, 1))), axis=1)
    D = np.array([0, 0, 1])
    1 == 1
    V = np.eye(2)
    x = eu.get_x0()
    x_hat = eu.get_x0_hat()

    n = 2
    for i in trange(total):
        A = eu.get_At(get_uncertainty)
        V, x_hat, (x, _) = alg.robustKalmanFilter(
            V, x_hat, radius, tol, A, BBT, C, DDT, BDT, n)
        np.linalg.norm(x_hat - x)
        all_v_x.append((V, x_hat.flatten))


if __name__ == '__main__':
    run(False, False, 1e-1, 1e-8, 1000)
