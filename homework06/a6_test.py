import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 4
K_VALUES = [0, 1, 5]

def Ax_columnblock(A, x, K, timings=None):
    matrix_n = comm.bcast(A.shape[0] if rank == 0 else None, root=0)
    cols_per_rank = matrix_n // size

    if matrix_n % size != 0:
        raise ValueError("matrix_n must be divisible by size")

    if K == 0:
        return x.copy() if rank == 0 else None

    if timings is not None:
        comm.Barrier()
        start = MPI.Wtime()

    local_A_T = np.empty((cols_per_rank, matrix_n), dtype=np.float64)

    if rank == 0:
        A_T = np.ascontiguousarray(A.T)
    else:
        A_T = None

    comm.Scatter(A_T, local_A_T, root=0)
    local_A = local_A_T.T

    # Distributing the x.
    local_x = np.empty(cols_per_rank, dtype=np.float64)

    comm.Scatter(
        x if rank == 0 else None,
        local_x,
        root=0
    )

    if timings is not None:
        comm.Barrier()
        kernel_start = MPI.Wtime()

    # Calculation.
    for step in range(K):
        local_y = local_A @ local_x

        if step < K - 1:
            comm.Reduce_scatter_block(
                local_y,
                local_x,
                op=MPI.SUM
            )
        else:
            if rank == 0:
                result = np.empty(matrix_n, dtype=np.float64)
            else:
                result = None

            comm.Reduce(local_y, result, op=MPI.SUM, root=0)

    if timings is not None:
        end = MPI.Wtime()
        setup_max = comm.reduce(kernel_start - start, op=MPI.MAX, root=0)
        kernel_max = comm.reduce(end - kernel_start, op=MPI.MAX, root=0)
        total_max = comm.reduce(end - start, op=MPI.MAX, root=0)
        if rank == 0:
            timings.update(setup=setup_max, kernel=kernel_max, end_to_end=total_max)

    return result if rank == 0 else None

    
matrices = {
    "identity": np.eye(N),
    "zero": np.zeros((N, N)),
    "diagonal": np.diag([1.0, 2.0, 3.0, 4.0]),
    "dense": np.arange(1, N * N + 1, dtype=np.float64).reshape(N, N) / 16,
}

if __name__ == "__main__":
    for name, matrix in matrices.items():
        for K in K_VALUES:
            A = matrix if rank == 0 else None
            x = np.arange(1, N + 1, dtype=np.float64) if rank == 0 else None
            y = Ax_columnblock(A, x, K)
            if rank == 0:
                expected = x.copy()
                for _ in range(K):
                    expected = A @ expected
                np.testing.assert_allclose(y, expected, rtol=1e-10, atol=1e-12)
                print(f"P={size}, K={K}, {name}: passed", flush=True)
