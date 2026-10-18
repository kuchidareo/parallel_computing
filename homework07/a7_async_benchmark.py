import numpy as np
import sys
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
rng = np.random.default_rng(2026 + rank)
A_rowbl = rng.random((n, N))
B_colbl = rng.random((N, n))

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
            print(f"{name}, trial={trial + 1}, total={values[0]:.6f}s, "
                  f"computation={values[1]:.6f}s, remainder={values[2]:.6f}s",
                  flush=True)
    if rank == 0:
        median = np.median(trials, axis=0)
        medians[name] = median[0]
        print(f"P={size}, N={N}, {name} median: total={median[0]:.6f}s, "
              f"computation={median[1]:.6f}s, remainder={median[2]:.6f}s",
              flush=True)

if rank == 0:
    print("Blocking / nonblocking median total:",
          medians["blocking"] / medians["nonblocking"], flush=True)
