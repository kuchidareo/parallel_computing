import numpy
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank_m = comm.Get_rank()
size_m = comm.Get_size()

def return_every_3rd_number(source, dest, tag, items):
    rank = comm.Get_rank()

    if rank == source:
        comm.Send(items, dest=dest, tag=tag)
        returned = numpy.empty_like(items[::3])
        comm.Recv(returned, source=dest, tag=tag + 1)
        return returned

    if rank == dest:
        status = MPI.Status()
        comm.Probe(source=source, tag=tag, status=status)
        count = status.Get_count(MPI.DOUBLE)
        received = numpy.empty(count, dtype=numpy.float64)
        comm.Recv(received, source=source, tag=tag)
        comm.Send(received[::3].copy(), dest=source, tag=tag + 1)
        return received

    return None

if size_m <= 1:
    print('Start at least 2 engines!')
else:
    source=0; dest=1
    items=numpy.arange(100,dtype=numpy.float64)
    data=return_every_3rd_number(source,dest,111,items)
    if rank_m == source:
        print('process',rank_m,'received:',data)
        numpy.testing.assert_equal(data,items[::3])
    elif rank_m == dest:
        print('process',rank_m,'received:',data)
        numpy.testing.assert_equal(data,items)
    else:
        numpy.testing.assert_equal(data,None)
