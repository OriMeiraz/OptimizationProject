import numpy as np
from utils import matrix_inner_product
from numpy.linalg import LinAlgError
MAX_ITERS = 5 * 10 ** 4
"""
Wasserstein Distributionally Robust Kalman Filtering, https://arxiv.org/pdf/1809.08830.pdf
Greatly based on the Matlab implementation by the paper's authors, found in https://github.com/sorooshafiee/WKF
"""


class WassersteinRobustKF:
    def __init__(self, wasserstein_radius, delta, state_size, observation_size, raise_errors=False):
        self.wasserstein_radius = wasserstein_radius
        self.delta = delta
        self.__gamma_eye_minus_D_inv_cache__ = dict()
        self.state_size = state_size
        self.observation_size = observation_size
        self.d = observation_size + state_size
        self.raise_errors = raise_errors

    def __gamma_eye_minus_D_inv__(self, gamma, P, P_inv, eigenvals):
        if gamma in self.__gamma_eye_minus_D_inv_cache__.keys():
            ans = self.__gamma_eye_minus_D_inv_cache__[gamma]
            return np.copy(ans)
        ans = P @ np.diag(1/(gamma - eigenvals)) @ P_inv
        ans = np.real(ans)
        "we have D = P @ diag(eigenvals) @ P^-1, therefore" \
        "gamma * I - D = P @ diag(gamma - eigenvals) @ P^-1, which implies" \
        "(gamma * I - D)^-1 = P @ diag( 1/(gamma - eigenvals) ) @ P^-1 "
        self.__gamma_eye_minus_D_inv_cache__[gamma] = ans
        return np.copy(ans)

    def bisection_alg(self, cov_matrix: np.ndarray, grad_matrix: np.ndarray, epsilon: float):
        """
        implements Alg 1 from the paper.
        :param epsilon: error tolerance. epsilon = epsilon(self.delta) = alpha_k * self.delta * C_upperline
        :param cov_matrix: $\Sigma >> 0$
        :param grad_matrix: $D >= 0$
        :return:
        """
        self.__gamma_eye_minus_D_inv_cache__ = dict()

        eigenvals, P = np.linalg.eig(grad_matrix)  # D = P diag(eigenvals) P^-1
        d = cov_matrix.shape[0]
        try:
            P_inv = np.linalg.inv(P)
        except LinAlgError:
            P_inv = np.linalg.inv(P + 1e-6 * np.eye(d))
            print('P is singular, added 1e-6*I')
        lambda_max_ind = np.argmax(eigenvals)
        lambda_1 = eigenvals[lambda_max_ind]
        v1 = P[:, lambda_max_ind]
        if d != self.d:
            raise ValueError(f'the size of Sigma ({d}) is not equal to the sum of the state size ({self.state_size})'
                             f' and the observation size ({self.observation_size})')
        LB = lambda_1 * (1 + np.sqrt(v1 @ (cov_matrix @ v1)) / self.wasserstein_radius)
        UB = lambda_1 * (1 + np.sqrt(np.trace(cov_matrix)) / self.wasserstein_radius)
        LB = np.real(LB)
        UB = np.real(UB)

        def h(gamma):
            gamma_eye_minus_D_inv = self.__gamma_eye_minus_D_inv__(gamma, P, P_inv, eigenvals)
            B_half = np.eye(d) - gamma * gamma_eye_minus_D_inv
            B = B_half @ B_half
            return self.wasserstein_radius ** 2 - matrix_inner_product(cov_matrix, B)

        stop_condition_met = False

        num_iters = 0
        h_vals = []
        gamma = 0
        while not stop_condition_met:
            num_iters += 1
            if UB == LB or num_iters > MAX_ITERS:
                # UB = UB.astype('float128')
                # LB = LB.astype('float128')
                message = f'after {num_iters} bisection iterations UB = {UB} == LB = ' \
                          f'{LB}, but stopping condition not met'
                if self.raise_errors:
                    raise RuntimeError(message)
                else:
                    print(message)
                    return L
            gamma = (UB + LB) / 2
            gamma_eye_minus_D_inv = self.__gamma_eye_minus_D_inv__(gamma, P, P_inv, eigenvals)
            L = gamma ** 2 * gamma_eye_minus_D_inv @ cov_matrix @ gamma_eye_minus_D_inv
            h_val = np.real(h(gamma))
            h_vals.append(h_val)
            if h_val < 0:
                LB = gamma
            else:
                UB = gamma

            delta = gamma * (self.wasserstein_radius ** 2 - np.trace(cov_matrix)) - \
                    matrix_inner_product(L, grad_matrix) + \
                    gamma ** 2 * matrix_inner_product(gamma_eye_minus_D_inv, cov_matrix)

            stop_condition_met = (delta < epsilon) and (h_val > 0)

        return L

    def frank_wolfe_alg(self, mu, cov_matrix: np.ndarray, iter_max=1000):
        """
        algorithm 2 in the paper, solves problem (5)
        :param mu: mean vector of hte prior distribution
        :param iter_max: maximal number of iteration before returning an estimate
        :param cov_matrix: covariance matrix of the prior distribution
        :return:
                 obj:            array of the objective values per iteration
                 res:            array of dual optimality gap in any iterations of Frank-Wolfe algorithm
                 Q_star_mu:      Mean vector of the least favorable prior
                 Q_star_Sigma:   Covariance matrix of the least favorable prior
                 phi_star_g:     Intercept of the optimal decision rule
                 phi_star_G:     Slope of the optimal decision rule
        """
        eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
        sigma_under = np.min(eigenvals)
        sigma_over = (self.wasserstein_radius + np.sqrt(np.trace(cov_matrix))) ** 2
        C_overbar = 2 * (sigma_over ** 4) / (sigma_under ** 3)
        k = 0
        S_k = cov_matrix
        stopping_condition_met = False
        obj = []
        res = []

        if self.wasserstein_radius == 0:  # return Bayes estimator
            Q_star_mu = mu
            Q_star_sigma = cov_matrix

            S_xy = S_k[:self.state_size, self.state_size:]  # the covariance between the state x and observation y
            S_yy = S_k[self.state_size:, self.state_size:]  # the covariance between the observation and itself
            G = S_xy @ np.linalg.inv(S_yy)
            phi_star_G = G
            phi_star_g = mu[: self.state_size] - G @ mu[self.state_size:]
            return obj, res, Q_star_mu, Q_star_sigma, phi_star_g, phi_star_G

        while k < iter_max and not stopping_condition_met:
            alpha_k = 2 / (k + 2)
            S_xy = S_k[:self.state_size, self.state_size:]  # the covariance between the state x and observation y
            S_yy = S_k[self.state_size:, self.state_size:]  # the covariance between the observation and itself
            S_xx = S_k[:self.state_size, :self.state_size]  # the covariance between the state and itself
            G = S_xy @ np.linalg.inv(S_yy)
            D = np.hstack([np.eye(self.state_size), -G]).T @ np.hstack([np.eye(self.state_size), -G])
            eps = alpha_k * self.delta * C_overbar
            L = self.bisection_alg(cov_matrix=cov_matrix, grad_matrix=D, epsilon=eps)

            current_res = abs((L - S_k).flatten() @ D.flatten())
            current_obj = np.trace(S_xx - G @ S_xy.T)

            stopping_condition_met = current_res / current_obj < self.delta

            if not stopping_condition_met:
                updated_S = (1 - alpha_k) * S_k + alpha_k * L
                S_k = updated_S
                k += 1
                obj.append(current_obj)
                res.append(current_res)

        Q_star_mu = mu
        Q_star_sigma = S_k
        phi_star_G = G
        phi_star_g = mu[:self.state_size] - G @ mu[self.state_size:]

        return obj, res, Q_star_mu, Q_star_sigma, phi_star_g, phi_star_G

    def find_estimate(self, A_t, B_t, C_t, D_t, observations, x0, V0, rho_t=None, **FW_kwargs):
        """

        :param rho_t:
        :param A_t: Matrix A_t in state-space representation
                    sys.A has to be a matrix with size (n * n) or
                    an array of matrices with length T with elements whose size is (n * n)
        :param B_t: same as A, just for B. size is (n * d)
        :param C_t: same as A, just for C. size is (m * n)
        :param D_t: same as A, just for D. size is (m * d)
        :param observations: Matrix of size (m * T), where the t'th observation y_t = observations[:, t]
        :param x0: initial estimate for the state
        :param V0: initial estimate for the covariance
        :param FW_kwargs: args for the Frank-Wolfe algorithm
        :return:
        """

        def normalize_A_t(mat):
            mat = np.array(mat)
            if mat.ndim == 3:
                get_mat = lambda t: mat[:, :, t]
            else:
                get_mat = lambda t: mat
            return mat, get_mat

        A_t, get_A = normalize_A_t(A_t)
        B_t, get_B = normalize_A_t(B_t)
        C_t, get_C = normalize_A_t(C_t)
        D_t, get_D = normalize_A_t(D_t)

        rho_t_passed = rho_t is not None
        if rho_t_passed:
            try:
                _ = iter(rho_t)
                get_rho_t = lambda t: rho_t[t]
            except TypeError:
                get_rho_t = lambda t: rho_t

        T = observations.shape[1]
        x_prev = x0
        V_prev = V0
        x_estimates = []
        V_estimates = []

        for t in range(T):
            A_aug = np.vstack([get_A(t), get_C(t) @ get_A(t)])
            B_aug = np.vstack([get_B(t), get_D(t) + get_C(t) @ get_B(t)])

            mu_before = A_aug @ x_prev
            cov_before = A_aug @ V_prev @ A_aug.T + B_aug @ B_aug.T

            if rho_t_passed:
                self.wasserstein_radius = get_rho_t(t)

            obj, res, Q_star_mu, Q_star_sigma, phi_star_g, phi_star_G \
                = self.frank_wolfe_alg(mu=mu_before, cov_matrix=cov_before, **FW_kwargs)

            V_t = Q_star_sigma[:self.state_size, :self.state_size] - \
                  phi_star_G @ Q_star_sigma[self.state_size:, :self.state_size]  # S_t,xx - G @ S_t,yx
            y_t = observations[:, t]
            x_hat = phi_star_G @ (y_t - Q_star_mu[self.state_size:]) + Q_star_mu[:self.state_size]

            x_estimates.append(x_hat)
            V_estimates.append(V_t)

            x_prev = x_hat
            V_prev = V_t

        return x_estimates, V_estimates


if __name__ == '__main__':
    pass
