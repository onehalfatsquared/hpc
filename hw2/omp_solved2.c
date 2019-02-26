/******************************************************************************
* FILE: omp_bug2.c
* DESCRIPTION:
*   Another OpenMP program with a bug. 
* AUTHOR: Blaise Barney 
* LAST REVISED: 04/06/05 
******************************************************************************/
/*****************************************************************************
 Bug Notes: There is a race condition present. Different result each time it runs/
            made tid private by defining in parallel region. Performed a reduction 
            on total to alleviate race condition. 
            There are also floating point errors which I fixed by using longs 
            instead of floats for total. 
********************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{
int nthreads;
//moved i and tid declaration into pragma omp so they are private
long total = 0; // float causes floating point errors in calculation. made long

/*** Spawn parallel region ***/
#pragma omp parallel 
  {
  /* Obtain thread number */
  int tid = omp_get_thread_num();
  /* Only master thread does this */
  if (tid == 0) {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d is starting...\n",tid);

  #pragma omp barrier

  /* do some work */
  #pragma omp for schedule(dynamic,10) reduction(+:total) //added reduction to compute total
  for (int i=0; i<1000000; i++) 
     total = total + i;

  printf ("Thread %d is done! Total= %ld\n",tid,total);

  } /*** End of parallel region ***/
}
