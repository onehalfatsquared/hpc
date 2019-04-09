#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>


void dp0(double* a, double* b, long N, double& dp) {
	//compute the dot product of a and b using the cpu

	double sum = 0;
	#pragma omp parallel for reduction(+:sum)
	for (long i = 0; i < N; i++) {
		sum += a[i]*b[i];
	}
}






















int main() {
  //long N = (1UL<<10); //10 was 25
  long N = 100;

  //initialize vector
  double *v;
  cudaMallocHost((void **) &v, N * sizeof(double));
  //#pragma omp parallel for 
  for (long i = 0; i < N; i++) {
  	printf("test\n");
    v[i] = 1.0/(i+1);
  }

  //initialize matrix
  double* a;
  cudaMallocHost((void **) &a, N*N*sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N*N; i++) {
  	a[i] = drand48(); 
  }

  //test dot product
  double dp, dp_ref;
  dp0(v,v,N,dp_ref);


  //double sum_ref, sum;
  //double tt = omp_get_wtime();
  //reduction(sum_ref, x, N);
  //printf("CPU Bandwidth = %f GB/s\n", 1*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  

  return 0;
}