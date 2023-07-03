import numpy as np
import exp2_utils as eu
import Algorithms as alg
from tqdm import trange, tqdm
import pandas as pd
import utils
import time
import os
import argparse
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm


C = np.array([1, -1])
B = np.array([[1.9608, 0.0195],
              [0.0195, 1.9605]])
B = np.concatenate((sqrtm(B), np.zeros((2, 1))), axis=1)
D = np.array([0, 0, 1])
DDT = np.array([1])
BDT = np.array([0, 0])
BBT = np.array([[1.9608, 0.0195],
                [0.0195, 1.9605]])


def run(time_var: bool, small: bool, radius, tol, total: int = 1000):
    all_norm_t = []
    get_uncertainty = eu.getUncertainty(time_var, small, total)

    V = np.eye(2)
    x = eu.get_x0()
    y = eu.get_y0()
    x_hat = eu.get_x0_hat()

    n = 2
    t0 = time.time()
    for _ in trange(total, leave=False):
        A = eu.get_At(get_uncertainty)
        x = A @ x + B @ np.random.randn(3)
        y = C @ x + D @ np.random.randn(3)
        V, x_hat = alg.robustKalmanFilter(
            V, x_hat, radius, tol, A, BBT, C, DDT, BDT, y, n)

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
    min_rad = np.linspace(0.1, 0.2, 11)[np.argmin(all_norms)]
    min_rad = round(min_rad, 2)
    path = f'Experiment2/saved_data/tv_{tv}__small_{small}__rad_{min_rad}__tol_{tol}'
    for i in trange(500, leave=False, desc=f'loading for best radius ({min_rad})'):
        df = pd.read_csv(
            f'{path}/exp_{i}.csv')
        all_norm.append(df['norm'].values)
    return min_rad, np.array(all_norm)


def load_data(tv, small, tol):
    return get_best_radius(tv, small, tol)


def run_experiment(tv, small, radius, tol):
    if tv == 'all':
        for tv in tqdm(['True', 'False'], leave=False, desc='running all tv values'):
            run_experiment(tv, small, radius, tol)
        return
    if small == 'all':
        for small in tqdm(['True', 'False'], leave=False, desc='running all small values'):
            run_experiment(tv, small, radius, tol)
        return

    path = f'Experiment2/saved_data/tv_{tv}__small_{small}__rad_{radius}__tol_{tol}'
    os.makedirs(path, exist_ok=True)
    for i in trange(500, leave=False, desc=f'running for radius {radius}'):
        df = run(tv, small, radius, tol)
        df.to_csv(f'{path}/exp_{i}.csv', index=False)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_var', type=str,
                        default='all', choices=['True', 'False', 'all'])
    parser.add_argument('--small', type=str, default='all',
                        choices=['True', 'False', 'all'])
    parser.add_argument('--radius', type=float, default=0)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--tol', type=float, default=1e-8)
    parser.add_argument('--run_exp', type=utils.str2bool, default=True)

    args = parser.parse_args()
    return args


def plot_data(tv, small, tol):
    # check if norms is saved, if not save it
    if not os.path.exists(f'norms.npy'):
        _, norms = load_data(tv, small, tol)
        np.save('norms.npy', norms)
    else:
        print('loading norms')
        norms = np.load('norms.npy')

    means = norms.mean(axis=0)
    means = eu.smooth(10*np.log10(means), 19)
    plt.semilogx(range(1, len(means)+1), means)
    plt.xlim([1, len(means)])
    plt.show()


if __name__ == '__main__':
    args = get_args()
    np.random.seed(args.seed)
    tv = args.time_var
    small = args.small
    radius = args.radius
    tol = args.tol
    run_exp = args.run_exp

    if radius <= 0:
        print(
            f'time_var: {tv}, small: {small}, radius: checking all, tol: {tol}')
    else:
        print(f'time_var: {tv}, small: {small}, radius: {radius}, tol: {tol}')

    if run_exp:
        if radius <= 0:
            for radius in tqdm(np.linspace(0.1, 0.2, 11), desc="running all radius"):
                radius = round(radius, 2)
                run_experiment(tv, small, radius, tol)
        else:
            run_experiment(tv, small, radius, tol)
    else:
        plot_data(tv, small, tol)
