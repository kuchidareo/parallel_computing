import numpy
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank_m = comm.Get_rank()
size_m = comm.Get_size()

def return_3rd_nonblocking(source, dest, tag, items):
    rank = comm.Get_rank()

    if rank == source:
        request = comm.isend(items, dest=dest, tag=tag)
        print('process', rank, 'starting first wait')
        request.wait()

        request = comm.irecv(source=dest, tag=tag + 1)
        print('process', rank, 'starting second wait')
        return request.wait()

    if rank == dest:
        request = comm.irecv(source=source, tag=tag)
        print('process', rank, 'starting first wait')
        received = request.wait()

        request = comm.isend(received[2], dest=source, tag=tag + 1)
        print('process', rank, 'starting second wait')
        request.wait()
        return received

    return None

if size_m <= 1:
    print('Start at least 2 engines!')
else:
    source=0; dest=1; items=[0.1,'Yes','Kolmas',0.001]
    data=return_3rd_nonblocking(source,dest,111,items)
    if rank_m == source:
        numpy.testing.assert_equal(data,'Kolmas')
    elif rank_m == dest:
        numpy.testing.assert_equal(data,items)
    else:
        numpy.testing.assert_equal(data,None)
