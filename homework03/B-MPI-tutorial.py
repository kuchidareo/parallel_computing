from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# sending message from 0 to 1.
if rank == 0:
    data = {"message": "hello", "number": 123}
    comm.send(data, dest=1)
elif rank == 1:
    data = comm.recv(source=0)
    print("1. recv:", data)

# Waiting for all processes to finish. 
comm.Barrier()


# Async communication.
if rank == 0:
    request = comm.isend("async hello", dest=1)
    request.wait()
elif rank == 1:
    request = comm.irecv(source=0)
    data = request.wait()
    print("2. irecv:", data)

comm.Barrier()

# Sending Object by using Send(), Recv().
if rank == 0:
    numbers = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    comm.Send([numbers, MPI.DOUBLE], dest=1)
elif rank == 1:
    # we need to set the memory space.
    numbers = np.empty(3, dtype=np.float64)
    comm.Recv([numbers, MPI.DOUBLE], source=0)
    print("3. Recv:", numbers)