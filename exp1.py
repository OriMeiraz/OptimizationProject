import numpy as np
import exp1_utils as eu
import Algorithms as alg
from tqdm import trange
import pandas as pd
import utils
from scipy.linalg import sqrtm
import pickle
import time
import os


def run(time_var: bool, small: bool, radius, tol, total: int = 1000):
    all_norm_t = []
    get_uncertainty = eu.getUncertainty(time_var, small, total)
    C = np.array([1, -1])
    DDT = np.array([1])
    BDT = np.array([0, 0])
    BBT = np.array([[1.9608, 0.0195],
                    [0.0195, 1.9605]])
    1 == 1
    V = np.eye(2)
    x = eu.get_x0()
    x_hat = eu.get_x0_hat()

    n = 2
    t0 = time.time()
    for _ in range(total):
        A = eu.get_At(get_uncertainty)
        V, x_hat, (x, _) = alg.robustKalmanFilter(
            V, x_hat, radius, tol, A, BBT, C, DDT, BDT, n)
        norm = np.linalg.norm(x_hat - x)**2
        all_norm_t.append((norm, time.time() - t0))
    return pd.DataFrame(all_norm_t, columns=['norm', 'time'])


if __name__ == '__main__':
    try:
        os.mkdir(
            f'Experiment1/saved_data/tv_{False}__small_{False}__rad_{1e-1}')
    except:
        pass
    for i in trange(500):
        df = run(False, False, 1e-1, 1e-8, 1000)
        # save df to saved_data
        df.to_csv(
            f'Experiment1/saved_data/tv_{False}__small_{False}__rad_{1e-1}/exp_{i}.csv')
