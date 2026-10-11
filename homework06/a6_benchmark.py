import json
import sys
import numpy as np
from mpi4py import MPI
from a4_test import Ax_rowblock
from a6_test import Ax_columnblock

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
K = int(sys.argv[2]) if len(sys.argv) > 2 else 50

if rank == 0:
    rng = np.random.default_rng(2026)
    A = rng.random((N, N)) / N
    x = rng.random(N)
    expected = x.copy()
    for _ in range(K):
        expected = A @ expected
else:
    A = x = None

for name, solver in [("Row", Ax_rowblock), ("Column", Ax_columnblock)]:
    y = solver(A, x, K)  # Warm-up and correctness check.
    if rank == 0:
        np.testing.assert_allclose(y, expected, rtol=1e-10, atol=1e-12)
    for run in range(5):
        timings = {}
        solver(A, x, K, timings)
        if rank == 0:
            print(json.dumps(dict(strategy=name, P=comm.Get_size(), run=run + 1,
                                  N=N, K=K, **timings)), flush=True)
