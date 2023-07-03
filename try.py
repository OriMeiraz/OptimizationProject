import numpy as np
import numpy.linalg as LA
import time
from torch import cholesky_inverse as inv

A = np.random.rand(3, 3)
A = A @ A.T

N = 100000
gamma = 0.123


t0 = time.time()
for _ in range(N):
    LA.matrix_power(A, 2)
print(time.time() - t0)

t1 = time.time()
n = A.shape[0]
I = np.identity(n)
for _ in range(N):
    A @ A
print(time.time() - t1)
