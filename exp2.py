import numpy as np
import exp2_utils as eu
import Algorithms as alg
from tqdm import trange, tqdm
import pandas as pd
import utils
import time
import os
import argparse


def run(time_var: bool, small: bool, radius, tol, total: int = 1000):
    all_norm_t = []
    get_uncertainty = eu.getUncertainty(time_var, small, total)
    C = np.array([1, -1])
    DDT = np.array([1])
    BDT = np.array([0, 0])
    BBT = np.array([[1.9608, 0.0195],
                    [0.0195, 1.9605]])
    V = np.eye(2)
    x = eu.get_x0()
    x_hat = eu.get_x0_hat()

    n = 2
    t0 = time.time()
    for i in trange(total, leave=False):
        A = eu.get_At(get_uncertainty)
        V, x_hat, (x, _) = alg.robustKalmanFilter(
            V, x_hat, radius, tol, A, BBT, C, DDT, BDT, n)
        norm = np.linalg.norm(x_hat - x)**2
        all_norm_t.append((norm, time.time() - t0))
    return pd.DataFrame(all_norm_t, columns=['norm', 'time'])


def get_best_radius(tv, small, tol):
    all_norms = []
    for radius in tqdm(np.linspace(0.1, 0.2, 11), desc="getting best radius", leave=False):
        all_norm_mean = []
        radius = round(radius, 2)
        path = f'Experiment2/saved_data/tv_{tv}__small_{small}__rad_{radius}__tol_{tol}'
        for i in trange(500, leave=False, desc=f'loading for radius {radius}'):
            df = pd.read_csv(
                f'{path}/exp_{i}.csv')
            all_norm_mean.append(df['norm'].mean())
        all_norms.append(np.mean(all_norm_mean))

    # get all the mean norms for the best radius
    all_norm = []
    for i in trange(500, leave=False, desc=f'loading for best radius ({radius})'):
        df = pd.read_csv(
            f'{path}/exp_{i}.csv')
        all_norm.append(df['norm'].values)
    return np.linspace(0.1, 0.2, 11)[np.argmin(all_norms)], np.array(all_norm)


def load_data(tv, small, tol):
    get_best_radius(tv, small, tol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_var', type=utils.str2bool, default=False)
    parser.add_argument('--small', type=utils.str2bool, default=False)
    parser.add_argument('--radius', type=float, default=1e-1)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--tol', type=float, default=1e-8)
    parser.add_argument('--run_exp', type=utils.str2bool, default=False)

    args = parser.parse_args()
    np.random.seed(args.seed)
    tv = args.time_var
    small = args.small
    radius = args.radius
    tol = args.tol
    run_exp = args.run_exp
    print(f'time_var: {tv}, small: {small}, radius: {radius}, tol: {tol}')

    path = f'Experiment2/saved_data/tv_{tv}__small_{small}__rad_{radius}__tol_{tol}'
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    if run_exp:
        for i in trange(500):
            df = run(tv, small, radius, tol)
            # save df to saved_data
            df.to_csv(
                f'{path}/exp_{i}.csv')
    else:
        load_data(tv, small, tol)
        # raise NotImplementedError("implement loading of data")
