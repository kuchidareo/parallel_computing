from statistics import median
from time import perf_counter
import os

import numpy as np
from threadpoolctl import threadpool_info

N = 2000
REPEATS = 5

rng = np.random.default_rng(2026)
A = rng.random((N, N))
B = rng.random((N, N))

warmup = A @ B
del warmup

times = []
for _ in range(REPEATS):
    start = perf_counter()
    C = A @ B
    times.append(perf_counter() - start)

print("requested threads:", os.getenv("OPENBLAS_NUM_THREADS"))
print("times:", times)
print("median:", median(times))
print("runtime libraries:", threadpool_info())



