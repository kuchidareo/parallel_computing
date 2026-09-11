// omp_get_num_threads()
// omp_set_num_threads()
// omp_get_thread_num()
// omp_get_wtime()

#include <stdio.h>
#include <omp.h>

int main(void)
{
    omp_set_num_threads(4);
    double start = omp_get_wtime();
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        printf("Hello from thread %d of %d\n",
               id, num_threads);
    }
    double end = omp_get_wtime();
    printf("Execution time: %f seconds\n", end - start);
    return 0;
}
