import json
import os
import platform

import mpi4py
import numpy as np
from mpi4py import MPI

from a4_test import Ax_rowblock


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 4096
K = 10
REPEATS = 5

if rank == 0:
    rng = np.random.default_rng(2026)
    A = rng.random((N, N), dtype=np.float64) / N
    x = rng.random(N, dtype=np.float64)
else:
    A = None
    x = np.empty(N, dtype=np.float64)

# Correctness check.
result = Ax_rowblock(A, x, K)
if rank == 0:
    expected = x.copy()
    for _ in range(K):
        expected = A @ expected
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)

# Warm-up.
Ax_rowblock(A, x, K, {})

kernel_times = []
end_to_end_times = []
for _ in range(REPEATS):
    timings = {}
    Ax_rowblock(A, x, K, timings)
    if rank == 0:
        kernel_times.append(timings["kernel"])
        end_to_end_times.append(timings["end_to_end"])

hosts = comm.gather(MPI.Get_processor_name(), root=0)
if rank == 0:
    print(json.dumps({
        "ranks": size,
        "N": N,
        "K": K,
        "kernel_times": kernel_times,
        "end_to_end_times": end_to_end_times,
        "host": platform.node(),
        "rank_hosts": hosts,
        "mpi": MPI.Get_library_version().replace("\x00", "").splitlines()[0],
        "mpi4py": mpi4py.__version__,
        "numpy": np.__version__,
        "threads": {
            name: os.environ.get(name)
            for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
        },
    }))
