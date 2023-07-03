import numpy as np
import time

A = np.random.rand(3, 3)
B = np.random.rand(3, 3)
N = 100000

t0 = time.time()
for _ in range(N):
    (A * B).sum()
print(time.time() - t0)

t1 = time.time()
for _ in range(N):
    np.trace(A.T @ B)
print(time.time() - t1)

print(A.dot(B))
