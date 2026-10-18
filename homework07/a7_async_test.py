import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 3 # number of rows (or columns) per process
N = size*n # (whole matrix size depending on #procs -- just here for the test)

# good for testing:
# A_rowbl=np.arange(n*N,dtype='d').reshape((n,N))+rank*n*N 
# B_colbl=np.arange(N*n,dtype='d').reshape((N,n))+rank*n*N

def ring_matrix_multiplication(A_rowbl, B_colbl):
    matrix_n = comm.bcast(A_rowbl.shape[1] if rank == 0 else None, root=0)
    n = matrix_n // size

    if matrix_n % size != 0:
        raise ValueError("matrix_n must be divisible by size")

    local_A = np.ascontiguousarray(A_rowbl, dtype=np.float64)
    local_B = np.ascontiguousarray(B_colbl, dtype=np.float64).copy()

    local_C = np.empty((n, matrix_n), dtype=np.float64)
    next_B = np.empty_like(local_B)

    # Calculation.
    for step in range(size):
        q = (rank - step) % size

        local_C[:, q * n:(q + 1) * n] = local_A @ local_B

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

    return result if rank == 0 else None

def ring_matrix_async_multiplication(A_rowbl, B_colbl):
    matrix_n = comm.bcast(A_rowbl.shape[1] if rank == 0 else None, root=0)
    n = matrix_n // size

    if matrix_n % size != 0:
        raise ValueError("matrix_n must be divisible by size")

    local_A = np.ascontiguousarray(A_rowbl, dtype=np.float64)
    local_B = np.ascontiguousarray(B_colbl, dtype=np.float64).copy()

    local_C = np.empty((n, matrix_n), dtype=np.float64)
    next_B = np.empty_like(local_B)

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

        local_C[:, q * n:(q + 1) * n] = local_A @ local_B

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

    return result if rank == 0 else None

for N in [size, 3 * size]:
    n = N // size
    dense = np.arange(1, N * N + 1, dtype=np.float64).reshape(N, N)
    rng = np.random.default_rng(2026)
    cases = [
        ("random", rng.random((N, N)), rng.random((N, N)), None),
    ]

    for name, A_test, B_test, expected in cases:
        A_rowbl = A_test[rank * n:(rank + 1) * n].copy()
        B_colbl = B_test[:, rank * n:(rank + 1) * n].copy()
        C_blocking = ring_matrix_multiplication(A_rowbl, B_colbl)
        C = ring_matrix_async_multiplication(A_rowbl, B_colbl)

        # Reconstruct the inputs on rank 0 for the sequential reference.
        B_colbl_transposed = B_colbl.transpose().copy()
        A = np.empty((N, N)) if rank == 0 else None
        B = np.empty((N, N)) if rank == 0 else None
        comm.Gather(A_rowbl, A, root=0)
        comm.Gather(B_colbl_transposed, B, root=0)

        error = None
        if rank == 0:
            B = B.transpose()
            C_ref = A @ B
            try:
                np.testing.assert_allclose(C, C_blocking, rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(C_blocking, C_ref, rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(C, C_ref, rtol=1e-10, atol=1e-12)
                if expected is not None:
                    np.testing.assert_allclose(C, expected, rtol=1e-10, atol=1e-12)
                print(f"P={size}, N={N}, {name}: blocking = nonblocking = NumPy: passed", flush=True)
            except AssertionError as exc:
                error = str(exc)
                print(f"P={size}, N={N}, {name}: FAILED\n{error}", flush=True)

        error = comm.bcast(error, root=0)
        if error is not None:
            raise AssertionError(error)
