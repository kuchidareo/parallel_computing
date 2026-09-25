import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N=64
K_VALUES = [0, 1, 5]

def Ax_rowblock(A, x, K, timings=None):
    matrix_n = comm.bcast(A.shape[0] if rank == 0 else None, root=0)
    rows_per_rank = matrix_n // size

    if K == 0:
        return x.copy() if rank == 0 else None

    if timings is not None:
        comm.Barrier()
        end_to_end_start = MPI.Wtime()

    # Distributing the independent rows of A.
    local_A = np.empty((rows_per_rank, matrix_n), dtype=np.float64)
    comm.Scatter(
        A if rank == 0 else None,
        local_A,
        root=0
    )

    # Distributing the x.
    if rank == 0:
        current_x = x.copy()
    else:
        current_x = np.empty(matrix_n, dtype=np.float64)
    comm.Bcast(current_x, root=0)

    if timings is not None:
        comm.Barrier()
        kernel_start = MPI.Wtime()

    # Calculation.
    for step in range(K):
        local_y = local_A @ current_x

        if step < K - 1:
            comm.Allgather(local_y, current_x)
        else:
            if rank == 0:
                result = np.empty(matrix_n, dtype=np.float64)
            else:
                result = None

            comm.Gather(local_y, result, root=0)

    if timings is not None:
        kernel_elapsed = MPI.Wtime() - kernel_start
        end_elapsed = MPI.Wtime() - end_to_end_start
        kernel_max = comm.reduce(kernel_elapsed, op=MPI.MAX, root=0)
        end_max = comm.reduce(end_elapsed, op=MPI.MAX, root=0)
        if rank == 0:
            timings["kernel"] = kernel_max
            timings["end_to_end"] = end_max

    return result if rank == 0 else None

def main():
    for K in K_VALUES:
        if rank == 0:
            rng = np.random.default_rng(2026)
            A = rng.random((N, N)) / N * 2
            x = rng.random(N)
            A_orig = A.copy()
            expected = x.copy()
        else:
            A = np.empty((N, N))
            x = np.empty(N)

        y = Ax_rowblock(A, x, K)

        if rank == 0:
            for _ in range(K):
                expected = A_orig @ expected
            np.testing.assert_allclose(y, expected, rtol=1e-10, atol=1e-12)
            print("Passed K =", K)


if __name__ == "__main__":
    main()
