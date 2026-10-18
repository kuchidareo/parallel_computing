import numpy as np
import sys
import json
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 3 # number of rows (or columns) per process
N = size*n # (whole matrix size depending on #procs -- just here for the test)

# good for testing:
# A_rowbl=np.arange(n*N,dtype='d').reshape((n,N))+rank*n*N 
# B_colbl=np.arange(N*n,dtype='d').reshape((N,n))+rank*n*N

def ring_matrix_multiplication(A_rowbl, B_colbl, timings=None):
    matrix_n = comm.bcast(A_rowbl.shape[1] if rank == 0 else None, root=0)
    n = matrix_n // size

    if matrix_n % size != 0:
        raise ValueError("matrix_n must be divisible by size")

    local_A = np.ascontiguousarray(A_rowbl, dtype=np.float64)
    local_B = np.ascontiguousarray(B_colbl, dtype=np.float64).copy()

    local_C = np.empty((n, matrix_n), dtype=np.float64)
    next_B = np.empty_like(local_B)

    computation = 0.0
    # Calculation.
    for step in range(size):
        q = (rank - step) % size

        start = MPI.Wtime()
        local_C[:, q * n:(q + 1) * n] = local_A @ local_B
        computation += MPI.Wtime() - start

        if step < size - 1:
            comm.Sendrecv(
                sendbuf=local_B,
                dest=(rank + 1) % size,
                recvbuf=next_B,
                source=(rank - 1) % size
            )

            local_B, next_B = next_B, local_B

    # Gather results.
    if rank == 0:
        result = np.empty((matrix_n, matrix_n), dtype=np.float64)
    else:
        result = None

    comm.Gather(local_C, result, root=0)

    if timings is not None:
        timings["computation"] = computation
    return result if rank == 0 else None

def ring_matrix_async_multiplication(A_rowbl, B_colbl, timings=None):
    matrix_n = comm.bcast(A_rowbl.shape[1] if rank == 0 else None, root=0)
    n = matrix_n // size

    if matrix_n % size != 0:
        raise ValueError("matrix_n must be divisible by size")

    local_A = np.ascontiguousarray(A_rowbl, dtype=np.float64)
    local_B = np.ascontiguousarray(B_colbl, dtype=np.float64).copy()

    local_C = np.empty((n, matrix_n), dtype=np.float64)
    next_B = np.empty_like(local_B)

    computation = 0.0
    # Calculation.
    for step in range(size):
        q = (rank - step) % size

        if step < size - 1:
            req_recv = comm.Irecv(
                next_B,
                source=(rank - 1) % size
            )
            req_send = comm.Isend(
                local_B,
                dest=(rank + 1) % size
            )

        start = MPI.Wtime()
        local_C[:, q * n:(q + 1) * n] = local_A @ local_B
        computation += MPI.Wtime() - start

        if step < size - 1:
            req_recv.Wait()
            req_send.Wait()

            local_B, next_B = next_B, local_B

    # Gather results.
    if rank == 0:
        result = np.empty((matrix_n, matrix_n), dtype=np.float64)
    else:
        result = None

    comm.Gather(local_C, result, root=0)

    if timings is not None:
        timings["computation"] = computation
    return result if rank == 0 else None

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
if N % size != 0:
    raise ValueError("N must be divisible by size")
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

medians = {}
for name, solver in [("blocking", ring_matrix_multiplication),
                     ("nonblocking", ring_matrix_async_multiplication)]:
    solver(A_rowbl, B_colbl)  # One warm-up.
    trials = []
    for trial in range(5):
        timings = {}
        comm.Barrier()
        start = MPI.Wtime()
        solver(A_rowbl, B_colbl, timings)
        total = MPI.Wtime() - start
        computation = timings["computation"]
        # Subtract on each rank before taking the maximum.
        values = np.empty(3) if rank == 0 else None
        comm.Reduce(np.array([total, computation, total - computation]),
                    values, op=MPI.MAX, root=0)
        if rank == 0:
            trials.append(values)
            print(json.dumps(dict(version=name, N=N, P=size, trial=trial + 1,
                                  total=values[0], computation=values[1],
                                  remainder=values[2])), flush=True)
    if rank == 0:
        median = np.median(trials, axis=0)
        medians[name] = median[0]
