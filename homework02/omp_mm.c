/******************************************************************************
* FILE: omp_mm.c
*
* https://hpc.llnl.gov/tuts/openMP/samples/C/omp_mm.c
*
* DESCRIPTION:  
*   OpenMp Example - Matrix Multiply - C Version
*   Demonstrates a matrix multiply using OpenMP. Threads share row iterations
*   according to a predefined chunk size.
* AUTHOR: Blaise Barney
* LAST REVISED: 06/28/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define NRA 62                 /* number of rows in matrix A */
#define NCA 15                 /* number of columns in matrix A */
#define NCB 7                  /* number of columns in matrix B */

int main(int argc, char *argv[])
{
  int i, j, k, run;
  int chunk = 10;
  int requested_threads = argc > 1 ? atoi(argv[1]) : 4;
  double a[NRA][NCA], b[NCA][NCB], c[NRA][NCB];
  double times[5];

  omp_set_num_threads(requested_threads);

  for (i = 0; i < NRA; i++)
    for (j = 0; j < NCA; j++)
      a[i][j] = i + j;

  for (i = 0; i < NCA; i++)
    for (j = 0; j < NCB; j++)
      b[i][j] = i * j;

  for (run = 0; run < 6; run++) {
    for (i = 0; i < NRA; i++)
      for (j = 0; j < NCB; j++)
        c[i][j] = 0;

    double start = omp_get_wtime();
    #pragma omp parallel for schedule(static, chunk) private(j, k)
    for (i = 0; i < NRA; i++)
      for (j = 0; j < NCB; j++)
        for (k = 0; k < NCA; k++)
          c[i][j] += a[i][k] * b[k][j];
    double elapsed = omp_get_wtime() - start;

    if (run > 0)
      times[run - 1] = elapsed;
  }

  for (i = 0; i < 4; i++)
    for (j = i + 1; j < 5; j++)
      if (times[j] < times[i]) {
        double temporary = times[i];
        times[i] = times[j];
        times[j] = temporary;
      }

  printf("Threads: %d\n", requested_threads);
  printf("Median elapsed time: %.9f seconds\n", times[2]);
  return 0;
}
