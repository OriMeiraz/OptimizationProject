import numpy as np
import exp1_utils as utils
print(utils.__file__)


if __name__ == '__main__':
    A = np.array([[0.9802, 0.0196+0.099],
                  [0, 0.9802]])
    BBT = np.array([[1.9608, 0.0195],
                    [0.0195, 1.9605]])
    C = np.array([1, -1])
    DDT = np.array([1])
    BDT = np.array([0, 0])
    utils.calc_v(A, BBT, C, DDT, BDT, np.eye(2))
