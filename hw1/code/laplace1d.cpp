// Compute a numerical solution to the 1-dimensional laplace equation using 
// Jacobi or Gauss Seidel iterations to solve the linear system. N is the number 
// of grid points and max_iter is max number of iterations

#include <stdio.h>
#include "utils.h"
#include <math.h>

double* applyA(int N, double* u) {
	//apply fd matrix for 2nd deriv to vector u
	double* U = (double*) malloc(N * sizeof(double)); // vector length N
	for (int i = 0; i < N; i++) {
		if (i == 0) {
			U[i] = (2*u[i] - u[i+1])*(N+1)*(N+1);
		}
		else if (i == N-1) {
			U[i] = (2*u[i] - u[i-1])*(N+1)*(N+1);
		}
		else {
			U[i] = (2*u[i] - u[i-1] - u[i+1])*(N+1)*(N+1);
		}
	}
	return U;
}

double computeRes(int N, double* u, double* f) {
	//compute the residual ||Au-f||_2

	//get the product Au
	double* C = applyA(N, u);
	double* r = (double*) malloc(N * sizeof(double)); // vector length N

	//compute Au-f 
	for (int i = 0; i < N; i++) r[i] = C[i] - f[i];
	free(C); 

	//compute 2 norm of r 
	double R = 0;                    //initialize sum
	for (int i = 0; i < N; i++) {
		R += r[i]*r[i];
	}
	R = sqrt(R);
	free(r); 
	return R;
}

double* jacobi(int N, double* f, int max_iter, double* guess) {
  //apply jacobi iteration to solve system

	//set parameters to iteration for fd matrix
	double A = 2*(N+1)*(N+1); double B = -1*(N+1)*(N+1);

	//initialize permanent storage
	double* u = (double*) malloc(N * sizeof(double)); // vector length N
	memcpy(u, guess, N * sizeof(double));

	//compute initial residual
	double res, res0 = computeRes(N, u, f);

	//plot header
	printf(" Iteration   Residual\n");

	//loop until max iterations or until tolerance reached
	for (int k = 0; k < max_iter; k++) {
		// print the iteration number and the residual
		printf("%10d %10f\n", k, res);
		//apply the iteration
		for (int i = 0; i < N; i++) {
			if (i == 0) {
				u[i] = 1/A*(f[i] - B*guess[i+1]);
			}
			else if (i == N-1) {
				u[i] = 1/A*(f[i] - B*guess[i-1]);
			}
			else {
				u[i] = 1/A*(f[i] - B*(guess[i-1] + guess[i+1])); 
			}
		}

		//compute the new residual
		res = computeRes(N, u, f); 

		//update guess
		memcpy(guess, u, N * sizeof(double));

		//check if tolerance is reached
		if (res0/res > 1e6) {
			break;
		}
	}
	return u;
}

double* gS(int N, double* f, int max_iter, double* guess) {
  //apply gauss-seidel iteration to solve system

	//set parameters to iteration for fd matrix
	double A = 2*(N+1)*(N+1); double B = -1*(N+1)*(N+1); 

	//compute initial residual 
	double res0 = computeRes(N, guess, f);
	double res = res0;

	//plot header
	printf(" Iteration   Residual\n");

	//loop until max iterations or until tolerance reached
	for (int k = 0; k < max_iter; k++) {
		// print the iteration number and the residual
		printf("%10d %10f\n", k, res);
		//apply the iteration
		for (int i = 0; i < N; i++) {
			if (i == 0) {
				guess[i] = 1/A*(f[i] - B*guess[i+1]);
			}
			else if (i == N-1) {
				guess[i] = 1/A*(f[i] - B*guess[i-1]);
			}
			else {
				guess[i] = 1/A*(f[i] - B*(guess[i-1] + guess[i+1])); 
			}
		}

		//compute the new residual
		res = computeRes(N, guess, f); 

		//check if tolerance is reached
		if (res0/res > 1e6) {
			break;
		}
	}
	return guess;
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
	double* u = (double*) malloc(N * sizeof(double)); // vector length N
	double* f = (double*) malloc(N * sizeof(double)); // vector length N

	//initialize initial guess to zeros and rhs to ones
	for (int i = 0; i < N; i++) u[i] = 0;
	for (int i = 0; i < N; i++) f[i] = 1;

	//apply the iterations, time
	Timer t;
  t.tic();
  //jacobi(N, f, max_iter, u);
  gS(N, f, max_iter, u);
  double time = t.toc();
  printf("Time taken: %3f seconds", time);

  //free the memory
  free(u); free(f);

  return 0;
}
