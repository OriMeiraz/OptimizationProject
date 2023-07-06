import numpy as np
from WKF import WKF
from scipy.linalg import sqrtm
from generate_data import generate_data
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import sys
import argparse
from rkalman import rkalman


def smooth(a, WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid')/WSZ
    r = np.arange(1, WSZ-1, 2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((start, out0, stop))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    print(v)
    raise argparse.ArgumentTypeError(f'Boolean value expected - got {v}.')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_var', type=str2bool,
                        default=True)
    parser.add_argument('--small', type=str2bool,
                        default=True)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--run_exp', type=str2bool, default=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    np.random.seed(args.seed)
    print(args)
    n = 2
    m = 1
    sys.A = np.array([[0.9802, 0.0196],
                      [0, 0.9802]])

    sys.C = np.array([1, -1])
    Q = np.array([[1.9608, 0.0195],
                  [0.0195, 1.9605]])

    R = 1
    sys.B = np.concatenate([sqrtm(Q), np.zeros((2, 1))], axis=1)
    sys.D = np.array([0, 0, 1])

    run_count = 500
    T = 10

    x_0 = np.array([0, 0])
    V_0 = np.eye(n)
    all_rho = [0.2]
    all_c = [2 * 1e-4]

    is_TV = args.time_var
    coeff = 0.99 if not args.small else 0.099
    tau = 0
    if args.run_exp:
        err_WKF = np.zeros((T, run_count, len(all_rho)))
        err_KF = np.zeros((T, run_count))
        err_KL = np.zeros((T, run_count, len(all_c)))
        for run in trange(run_count, desc='running experiments'):
            x, y, y0 = generate_data(sys, x_0, T, coeff, is_TV)
            xhat_kalman, _, _, _ = WKF(sys, 0, y,  x_0, V_0)
            err_KF[:, run] = np.sum((x - xhat_kalman)**2, axis=0)

            for k, rho in enumerate(all_rho):
                xhat, _, _, _ = WKF(sys, rho, y, x_0, V_0)
                err_WKF[:, run, k] = np.sum((x - xhat)**2, axis=0)

            y_delay = np.append([y0], y[:, :-1])
            for k, c in enumerate(all_c):
                xhat, _, _, _ = rkalman(sys, c, tau, y_delay, x_0, V_0)
                err_KL[:, run, k] = np.sum((x - xhat)**2, axis=0)

        np.save(f'err_KF_{args.time_var, args.small}.npy', err_KF)
        np.save(f'err_WKF_{args.time_var, args.small}.npy', err_WKF)

    else:
        try:
            err_KF = np.load(f'err_KF_{args.time_var, args.small}.npy')
            err_WKF = np.load(f'err_WKF_{args.time_var, args.small}.npy')
        except FileNotFoundError:
            print('Run the experiment first.')
            sys.exit()

    tmp = np.mean(err_WKF, axis=(0, 1))
    k_rho = np.argmin(tmp)
    print(k_rho)
    means_W = np.mean(err_WKF[:, :, k_rho], axis=1)
    means_W = smooth(10 * np.log10(means_W),  19)
    plt.semilogx(range(1, T+1), means_W, label='WKF')

    means = np.mean(err_KF, axis=1)
    means = smooth(10 * np.log10(means), 19)
    plt.semilogx(range(1, T+1), means, label='KF')

    plt.legend()
    plt.savefig(f'figure5_{args.time_var, args.small}.png')
    plt.show()
