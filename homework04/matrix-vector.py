import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

A = np.array([[1.0, 2.0], [3.0, 4.0]])
x = np.array([1.0, 1.0])

if rank == 0:
    original_x = x.copy()
    for worker in range(1, size):
        comm.send(x, dest=worker)
        x = comm.recv(source=worker)

    expected = np.linalg.matrix_power(A, size - 1) @ original_x
    np.testing.assert_allclose(x, expected)
    print("Result:", x)
else:
    x = comm.recv(source=0)
    x = A @ x
    comm.send(x, dest=0)
