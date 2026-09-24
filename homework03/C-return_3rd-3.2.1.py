import numpy
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank_m = comm.Get_rank()
size_m = comm.Get_size()

def return_3rd(source, dest, tag, items):
    rank = comm.Get_rank()
    if rank == source:
        comm.send(items, dest=dest, tag=tag)
        return comm.recv(source=dest, tag=tag + 1)
    if rank == dest:
        received = comm.recv(source=source, tag=tag)
        comm.send(received[2], dest=source, tag=tag + 1)
        return received
    return None

if size_m <= 1:
    print('Start at least 2 engines!')
else:
    source=0; dest=1; items=[0.1,'Yes','Kolmas',0.001]
    data=return_3rd(source,dest,111,items)
    if rank_m == source:
        print('process',rank_m,'received:',data)
        numpy.testing.assert_equal(data,'Kolmas')
    elif rank_m == dest:
        print('process',rank_m,'received:',data)
        numpy.testing.assert_equal(data,items)
    else:
        numpy.testing.assert_equal(data,None)
