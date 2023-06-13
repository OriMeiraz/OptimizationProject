"""
recreates experiment 1, described in section 5.1 of the paper (https://arxiv.org/pdf/1809.08830.pdf)
"""
import numpy as np
from frank_wolfe_alg import WassersteinRobustKF
from time import time
import matplotlib.pyplot as plt
import seaborn as sns


def generate_cov_matrices(d):
    """
    generates two d-dimensional covariance matrices, Sigma and Sigma_star such that:
        P = N_d (0, Sigma) ; P_star = N_d (0, Sigma_star)
        then
            W_2 (P, P_star) <= || Sigma^(0.5) - Sigma_star^(0.5) ||_F <= d^(0.5)

    :param d: the dimension of the covariance matrices. d = m + n = len(observation_vector) + len(state_vector)
    :return: Sigma, Sigma_star
    """
    A = np.random.randn(d, d)
    A_star = np.random.randn(d, d)

    R = np.linalg.eigh(A + A.T)[1]
    assert np.isclose(R @ R.T, np.eye(d)).all()

    R_star = np.linalg.eigh(A_star + A_star.T)[1]
    assert np.isclose(R_star @ R_star.T, np.eye(d)).all()

    # diagonal matrix with the diagonal ~ Uni[0,1)
    Lambda = np.diag(np.random.rand(d))
    # diagonal matrix with the diagonal ~ Uni[0.1,10)
    Lambda_star = np.diag(0.1 + 9.9 * np.random.rand(d))

    Sigma = R @ Lambda @ R.T
    Sigma_star_half = R @ np.sqrt(Lambda) @ R.T + \
        R_star @ np.sqrt(Lambda_star) @ R_star.T
    Sigma_star = Sigma_star_half @ Sigma_star_half

    assert np.linalg.norm(R_star @ np.sqrt(Lambda_star) @ R_star.T,
                          ord='fro') <= np.sqrt(10*d)  # an error in the paper
    return Sigma, Sigma_star


def single_experiment(d, state_size=0.8):
    Sigma, Sigma_star = generate_cov_matrices(d)
    if isinstance(state_size, float):
        state_size = int(d * state_size)
    robust_kf = WassersteinRobustKF(wasserstein_radius=np.sqrt(10*d), delta=1e-4, state_size=state_size,
                                    observation_size=d - state_size)
    time_start = time()
    temp = robust_kf.frank_wolfe_alg(mu=np.zeros(d), cov_matrix=Sigma_star)
    time_end = time()
    obj, res = temp[0], temp[1]
    return time_end - time_start, obj, res


def plot_4ab(num_exp=10**3, d=10, state_size=0.8):
    times = []
    for i in range(num_exp):
        time_delta, obj, res = single_experiment(d, state_size)
        times.append(time_delta)
        if (i % num_exp/20) == num_exp/20-1:
            print(f'finished {i+1} out of {num_exp} experiments')
    # plt.hist(times, density=True, bins=50)
    plt.title(f'run time of single Frank-Wolfe run, d={d}')
    plt.xlabel('time [s]')
    # plt.ylabel('frequency')
    # plt.show()
    fig = sns.histplot(times, stat='density', binwidth=0.1)
    # fig.set_axis_labels('time [s]', 'frequency')
    plt.show()
    # plt.savefig(f'plots/experiment 1/d_{d}_using_sns.png')
    fig.get_figure().savefig(f'plots/experiment 1/sns_d_{d}.png')


if __name__ == '__main__':
    for d in [10, 50, 100]:
        plot_4ab(d=d, num_exp=1000)
