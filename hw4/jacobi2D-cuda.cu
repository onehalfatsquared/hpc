// Compute a numerical solution to the 2-dimensional laplace equation using 
// Jacobi or Gauss Seidel iterations to solve the linear system. N is the number 
// of grid points per dimension and max_iter is max number of iterations

#include <stdio.h>
#include "utils.h"
#include <math.h>
#ifdef _OPENMP
	#include <omp.h>
#endif

#define BLOCK_SIZE 1024

int toIndex(int row, int column, int N) {
	//maps the (i,j) entry of u or f to 1d array index
	return N*row+column;
}

void index2ij(int index, int N, int& i, int& j) {
  // map a row index 1-d array into (i,j) 
  j = index % N;
  i = index / N;
}

void setDirections(int i, int j, int N, double* vec, double& left, double& right,
																											double& up, double& down) {
	//set the values of u at left, right, up, and down of current point (i,j)
	//handle the case where point is at bottom row
	if (i == 0) {
		down = 0; up = vec[toIndex(i+1, j, N)];
		if (j == 0) {
			left = 0; right = vec[toIndex(i, j+1, N)];
		}
		else if (j == N-1) {
			right = 0; 
			left = vec[toIndex(i, j-1, N)];
		}
		else {
			left = vec[toIndex(i, j-1, N)]; 
			right = vec[toIndex(i, j+1, N)];
		}
	}
	//handle case where point is at top row
	else if (i == N-1) {
		up = 0; down = vec[toIndex(i-1, j, N)];
		if (j == 0) {
			left = 0; right = vec[toIndex(i, j+1, N)];
		}
		else if (j == N-1) {
			right = 0; left = vec[toIndex(i, j-1, N)];
		}
		else {
			left = vec[toIndex(i, j-1, N)]; right = vec[toIndex(i, j+1, N)];
		}
	}
	//handle middle case
	else {
		down = vec[toIndex(i-1, j, N)]; up = vec[toIndex(i+1, j, N)];
		left = vec[toIndex(i, j-1, N)]; right = vec[toIndex(i, j+1, N)];
	}
}

double* jacobi(int N, double* f, int max_iter, double* guess) {
  //apply jacobi iteration to solve system

  //initialize storage to do update - left, right, up, and down of (i,j)
  double left, right, up, down; 
  int i, j; 

	//initialize permanent storage
	double* u = (double*) malloc(N * N * sizeof(double)); // vector length N^2
	memcpy(u, guess, N * N * sizeof(double));

	//set the interval length squared for use in iteration
	double H = 1.0 / ((N+1) * (N+1)); //H = h^2

	//loop until max iterations or until tolerance reached
	for (int k = 0; k < max_iter; k++) {
		//apply the iteration
		for (int point = 0; point < N * N; point++) {
			//get the (i,j) index of the current point
			index2ij(point, N, i, j);

			//set left, right, up, and down values
			setDirections(i, j, N, guess, left, right, up, down);

			//apply the jacobi update formula
			u[point] = 1.0/4.0*(H*f[point] + up + down + left + right);
		}
		//update guess
		memcpy(guess, u, N * N * sizeof(double));
	}

	//free the guess memory and return the solution
	free(guess);
	return u;
}

double* jacobiP(int N, double* f, int max_iter, double* guess) {
  //apply jacobi iteration to solve system - parallel

  //initialize storage to do update - left, right, up, and down of (i,j)
  double left, right, up, down; 
  int i, j; 

	//initialize permanent storage
	double* u = (double*) malloc(N * N * sizeof(double)); // vector length N^2
	memcpy(u, guess, N * N * sizeof(double));

	//set the interval length squared for use in iteration
	double H = 1.0 / ((N+1) * (N+1)); //H = h^2

	//loop until max iterations or until tolerance reached
	for (int k = 0; k < max_iter; k++) {
		//apply the iteration
		#pragma omp parallel for private(i, j, left, up, right, down)
		for (int point = 0; point < N * N; point++) {
			//get the (i,j) index of the current point
			index2ij(point, N, i, j);

			//set left, right, up, and down values
			setDirections(i, j, N, guess, left, right, up, down);

			//apply the jacobi update formula
			u[point] = 1.0/4.0*(H*f[point] + up + down + left + right);
		}
		//update guess
		memcpy(guess, u, N * N * sizeof(double));
	}

	//free the guess memory and return the solution
	free(guess);
	return u;
}

double* applyA(int N, double* u) {
	//apply fd matrix for 2nd deriv to vector u

	//initialize storage to do update - left, right, up, and down of (i,j)
  double left, right, up, down; 
  int i, j; 

	//set the interval length squared for use in iteration
	double H = 1.0 / ((N+1) * (N+1)); //H = h^2

	double* U = (double*) malloc(N * N * sizeof(double)); // vector length N
	for (int point = 0; point < N*N; point++) {
		//get the (i,j) index of the current point
		index2ij(point, N, i, j);

		//set left, right, up, and down values
		setDirections(i, j, N, u, left, right, up, down);

		//apply the jacobi update formula
		U[point] = 1/H * (4*u[point] - left - up - down - right);
	}
	return U;
}

double computeRes(int N, double* u, double* f) {
	//compute the residual ||Au-f||_2

	//get the product Au
	double* C = applyA(N, u);
	double* r = (double*) malloc(N * N * sizeof(double)); // vector length N

	//compute Au-f 
	for (int i = 0; i < N * N; i++) r[i] = C[i] - f[i];
	free(C); 

	//compute 2 norm of r 
	double R = 0;                    //initialize sum
	for (int i = 0; i < N * N; i++) {
		R += r[i]*r[i];
	}
	R = sqrt(R);
	free(r); 
	return R;
}

__global__ void JacobiKernel(int N, double* u, double* f, double* guess, double H) {
	//kernel to do jacobi update

	long row = blockIdx.x * blockDim.x + threadIdx.x;
	long col = blockIdx.y * blockDim.y + threadIdx.y;

	double left, right, up, down;

	if (row < N && col < N) {
		if (row == 0) 
			up = 0; 
		else 
			up = guess[N*col+row-1];
		if (row == N-1) 
			down = 0; 
		else 
			down = guess[N*col+row+1];
		if (col == 0) 
			left = 0;
		else 
			left = guess[N*(col-1)+row];
		if (col == N-1) 
			right = 0; 
		else 
			right = guess[N*(col+1)+row];

		u[N*col+row] = 0.25*(H*f[N*col+row] + up + down +left +right);
	}
}




void jacobiG(int N, double* f, int max_iter, double* guess) {
	//apply jacobi iteration on the gpu

	//initialize update 
	double *u_d, *f_d, *guess_d;
  cudaMalloc(&u_d, N*N*sizeof(double));
  cudaMalloc(&f_d, N*sizeof(double));
  cudaMalloc(&guess_d, N*N*sizeof(double));
  printf("cuda malloc done\n");

  cudaMemcpyAsync(u_d, guess, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(guess_d, guess, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(f_d, f, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

	//set the interval length squared for use in iteration
	double H = 1.0 / ((N+1) * (N+1)); //H = h^2

	for (int k = 0; k < max_iter; k++) {
		//call the kernel
		JacobiKernel<<<N/BLOCK_SIZE+1,BLOCK_SIZE>>>(N, u_d, f_d, guess_d, H);

		//copy u into guess
		cudaMemcpyAsync(guess_d, u_d, N*N*sizeof(double), cudaMemcpyDeviceToDevice);
  	cudaDeviceSynchronize();
	}

	//copy guess back to host
	cudaMemcpyAsync(guess, guess_d, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  //free cuda memory
  cudaFree(u_d); cudaFree(f_d); cudaFree(guess_d);
}


int main(int argc, char** argv) {
  //setup and call the solvers

	//handle input
	if (argc != 3) {
		fprintf(stderr, "Usage: %s <N> <max_iter>\n", argv[0]);
		return EXIT_FAILURE;
	}
  int N = atoi(argv[1]);
  int max_iter = atoi(argv[2]);

  // allocate memory for solution and rhs. 
	double* u = (double*) malloc(N * N * sizeof(double)); // vector length N^2
	double* f = (double*) malloc(N * N * sizeof(double)); // vector length N^2
	double* sol = (double*) malloc(N * N * sizeof(double)); // vector length N^2

	//initialize initial guess to zeros and rhs to ones
	for (int i = 0; i < N*N; i++) u[i] = 0;
	for (int i = 0; i < N*N; i++) f[i] = 1;

	//apply the iterations, time
	Timer t;
  t.tic();
  sol = jacobiP(N, f, max_iter, u);
  double time = t.toc();
  printf("CPU Time taken: %3f seconds\n", time);

  //compute a residual as a check
  double r = computeRes(N, sol, f);
  printf("CPU Residual : %3f\n", r);

  //free u
  free(u);

  //initialize initial guess to zeros and rhs to ones for gpu
  //double *u;
  cudaMalloc(&u, N*N*sizeof(double));
	for (int i = 0; i < N*N; i++) u[i] = 0;
	for (int i = 0; i < N*N; i++) f[i] = 1;

		for (int i = 0; i < N*N; i++) printf("%f\n", u[i]);

  //do on gpu
  //apply the iterations, time
  t.tic();
  jacobiG(N, f, max_iter, u);
  time = t.toc();
  printf("GPU Time taken: %3f seconds\n", time);

  //compute a residual as a check
  r = computeRes(N, u, f);
  printf("GPU Residual : %3f\n", r);

  for (int i = 0; i < N*N; i++) printf("%f\n", u[i]);

  //free the memory
  free(f); 

  return 0;
}
