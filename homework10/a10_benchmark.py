import json
import sys
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# Blocking ring multiplication from Homework07.
def ring_matrix_multiplication(A_rowbl, B_colbl):
    N = A_rowbl.shape[1]
    n = N // size
    local_B = B_colbl.copy()
    next_B = np.empty_like(local_B)
    local_C = np.empty((n, N))
    for step in range(size):
        q = (rank - step) % size
        local_C[:, q*n:(q+1)*n] = A_rowbl @ local_B
        if step < size - 1:
            comm.Sendrecv(local_B, dest=(rank+1) % size,
                          recvbuf=next_B, source=(rank-1) % size)
            local_B, next_B = next_B, local_B
    result = np.empty((N, N)) if rank == 0 else None
    comm.Gather(local_C, result, root=0)
    return result


N = int(sys.argv[1])
if N % size != 0:
    raise ValueError("N must be divisible by P")
n = N // size
if rank == 0:
    rng = np.random.default_rng(2026)
    A = rng.random((N, N))
    B = rng.random((N, N))
    B_blocks = np.stack([B[:, j*n:(j+1)*n] for j in range(size)])
else:
    A = B_blocks = None
A_rowbl = np.empty((n, N))
B_colbl = np.empty((N, n))
comm.Scatter(A, A_rowbl, root=0)
comm.Scatter(B_blocks, B_colbl, root=0)
del A, B_blocks
if rank == 0:
    del B

ring_matrix_multiplication(A_rowbl, B_colbl)  # One warm-up.
for trial in range(5):
    comm.Barrier()
    start = MPI.Wtime()
    ring_matrix_multiplication(A_rowbl, B_colbl)
    elapsed = MPI.Wtime() - start
    total = comm.reduce(elapsed, op=MPI.MAX, root=0)
    if rank == 0:
        print(json.dumps(dict(N=N, P=size, trial=trial+1, total=total)), flush=True)
