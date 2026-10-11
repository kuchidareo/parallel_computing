import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 12
K_VALUES = [0, 1, 5]

def Ax_cellblock(A, x, K):
    matrix_n = comm.bcast(A.shape[0] if rank == 0 else None, root=0)

    p = int(np.sqrt(size))
    if p * p != size:
        raise ValueError("Number of processes must be a perfect square")

    block_size = matrix_n // p

    i = rank // p
    j = rank % p

    row_comm = comm.Split(color=i, key=j)
    col_comm = comm.Split(color=j, key=i)

    if matrix_n % p != 0:
        raise ValueError("matrix_n must be divisible by p")

    if K == 0:
        return x.copy() if rank == 0 else None

    # Position of each process in the 2D grid.
    i = rank // p
    j = rank % p

    # Distributing the blocks of A. np.stack -> Scatter.
    local_A = np.empty((block_size, block_size), dtype=np.float64)

    if rank == 0:
        blocks = []
        for row in range(p):
            for col in range(p):
                block = A[
                    row*block_size:(row+1)*block_size,
                    col*block_size:(col+1)*block_size
                ]
                blocks.append(block)

        send_A = np.stack(blocks)
    else:
        send_A = None

    comm.Scatter(send_A, local_A, root=0)

    # Distributing the x. 
    local_x = np.empty(block_size, dtype=np.float64)

    if rank == 0:
        blocks_x = []

        for r in range(size):
            col_idx = r % p
            block = x[col_idx*block_size:(col_idx+1)*block_size]
            blocks_x.append(block)

        send_x = np.stack(blocks_x)
    else:
        send_x = None

    comm.Scatter(send_x, local_x, root=0)

    # Calculation.
    row_y = np.empty(block_size, dtype=np.float64)
    result = None

    for step in range(K):
        local_y = local_A @ local_x

        # Sum contributions within each row.
        row_comm.Allreduce(local_y, row_y, op=MPI.SUM)

        if step < K - 1:
            # Prepare x for the next iteration.
            if i == j:
                local_x[:] = row_y

            col_comm.Bcast(local_x, root=j)

        else:
            # Collect the final result on process 0.
            if j == 0:
                if rank == 0:
                    result = np.empty(matrix_n, dtype=np.float64)

                col_comm.Gather(row_y, result, root=0)

    return result if rank == 0 else None

    
matrices = {
    "identity": np.eye(N),
    "zero": np.zeros((N, N)),
    "diagonal": np.diag(np.arange(1, N + 1, dtype=np.float64)),
    "dense": np.arange(1, N * N + 1, dtype=np.float64).reshape(N, N) / (N * N),
}

if __name__ == "__main__":
    for name, matrix in matrices.items():
        for K in K_VALUES:
            A = matrix if rank == 0 else None
            x = np.arange(1, N + 1, dtype=np.float64) if rank == 0 else None
            y = Ax_cellblock(A, x, K)
            if rank == 0:
                expected = x.copy()
                for _ in range(K):
                    expected = A @ expected
                np.testing.assert_allclose(y, expected, rtol=1e-10, atol=1e-12)
                print(f"P={size}, K={K}, {name}: passed", flush=True)
