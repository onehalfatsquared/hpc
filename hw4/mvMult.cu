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

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

#define BLOCK_SIZE 1024

__global__ void reduction_kernel2(double* sum, const double* a, long N){
	//reduction kernel for summing
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}

__global__ void mult_kernel(double* a, double* b, double* c, long N) {
	//cuda kernel to compute pairwise multiplication of a and b
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		c[idx] = a[idx] * b[idx];
	}
}


void dot(double* a, double* b, long N, double& dp) {
	//take in a and b vectors, apply dot product and reduction kernels. 

	//allocate a vector for the product
	double *c_d;
	cudaMalloc(&c_d, N*sizeof(double));

	//call the multiplication kernel
	mult_kernel<<<N/BLOCK_SIZE+1,BLOCK_SIZE>>>(a, b, c_d, N);

	//call reduction kernel on c
	double *y_d;
  cudaMalloc(&y_d, ((N+BLOCK_SIZE-1)/BLOCK_SIZE)*sizeof(double));
	double* sum_d = y_d;
  long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
  reduction_kernel2<<<Nb,BLOCK_SIZE>>>(sum_d, c_d, N);
  while (Nb > 1) {
    long N = Nb;
    Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
    reduction_kernel2<<<Nb,BLOCK_SIZE>>>(sum_d + Nb, sum_d, N);
    sum_d += Nb;
  }


  cudaMemcpyAsync(&dp, sum_d, 1*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();





	//free memory
	cudaFree(c_d);
}



















int main() {
  long N = (1UL<<10); //10 was 25
  //long N = 100;

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

  //get a reference solution for dot product
  double dp;
  double dp_ref;
  dp0(v,v,N,dp_ref);

  //get a reference solution for matrix vector product
  double* mult;
  double* mult_ref;
  cudaMallocHost((void **) &mult_ref, N * sizeof(double));
  //mv0(a, v, N, mult_ref);

  //copy memory to gpu
  double *v_d, *a_d, *mult_d;
  cudaMalloc(&v_d, N*sizeof(double));
  cudaMalloc(&a_d, N*N*sizeof(double));
  cudaMalloc(&mult_d, N*sizeof(double));

  cudaMemcpyAsync(v_d, v, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(a_d, a, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  //do dot product on gpu
  dot(v_d, v_d, N, dp);

  //get error
  double errDP = fabs(dp_ref-dp);
  printf("Dot product Error: %f", errDP);


  //free memory
  cudaFree(v_d);
  cudaFree(a_d);
  cudaFreeHost(v);
  cudaFreeHost(a);
  cudaFreeHost(mult_ref); cudaFreeHost(mult);

  

  return 0;
}