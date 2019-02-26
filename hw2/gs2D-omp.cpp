// Compute a numerical solution to the 2-dimensional laplace equation using 
//  Gauss Seidel iterations to solve the linear system. N is the number 
// of grid points per dimension and max_iter is max number of iterations

#include <stdio.h>
#include "utils.h"
#include <math.h>
#ifdef _OPENMP
	#include <omp.h>
#endif


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

void getColor(int N, int* red, int* black) {
	//store index of points who are red and black
	int i, j; //to get indices
	int r = 0; int b = 0; //icrementers for re and black loops
	for (int point = 0; point < N*N; point++) {
  	index2ij(point, N, i, j);
  	if ((i+j) % 2 == 0) {
  		//node is red
  		red[r] = point; r++;
  	}
  	else{
  		//node is black
  		black[b] = point; b++;
  	}
  }
}

void gs(int N, double* f, int max_iter, double* guess) {
  //apply gs iteration to solve system - 2 colors


  //initialize storage to do update - left, right, up, and down of (i,j)
  double left, right, up, down; 
  int i, j; 

  //initialize red and black storage - compute entries
  int Np = (N*N+1)/2; int Nm = N*N/2; 
  int* red = (int*) malloc(Np * sizeof(int)); // vector length ceil(N^2/2)
  int* black = (int*) malloc(Nm * sizeof(int)); // vector length floor(N^2/2)
  getColor(N, red, black); 

	//set the interval length squared for use in iteration
	double H = 1.0 / ((N+1) * (N+1)); //H = h^2

	//loop until max iterations or until tolerance reached
	for (int k = 0; k < max_iter; k++) {
		//apply the iteration
		//loop over red points
		for (int rp = 0; rp < Np; rp++) {
			//get the (i,j) index of the current point
			int point = red[rp]; 
			index2ij(point, N, i, j);

			//set left, right, up, and down values
			setDirections(i, j, N, guess, left, right, up, down);

			//apply the update formula
			guess[point] = 1.0/4.0*(H*f[point] + up + down + left + right);
		}
		//loop over black points
		for (int bp = 0; bp < Nm; bp++) {
			//get the (i,j) index of the current point
			int point = black[bp];
			index2ij(point, N, i, j);

			//set left, right, up, and down values
			setDirections(i, j, N, guess, left, right, up, down);

			//apply the update formula
			guess[point] = 1.0/4.0*(H*f[point] + up + down + left + right);
		}
	}

	//free black and red
	free(black); free(red); 
}


void gsP(int N, double* f, int max_iter, double* guess) {
  //apply gs iteration to solve system - parallel - 2 colors

  //initialize storage to do update - left, right, up, and down of (i,j)
  double left, right, up, down; 
  int i, j; 

  //initialize red and black storage - compute entries
  int Np = (N*N+1)/2; int Nm = N*N/2; 
  int* red = (int*) malloc(Np * sizeof(int)); // vector length ceil(N^2/2)
  int* black = (int*) malloc(Nm * sizeof(int)); // vector length floor(N^2/2)
  getColor(N, red, black); 

	//set the interval length squared for use in iteration
	double H = 1.0 / ((N+1) * (N+1)); //H = h^2

	//loop until max iterations or until tolerance reached
	for (int k = 0; k < max_iter; k++) {
		//apply the iteration
		//loop over red points
		#pragma omp parallel 
		{
		#pragma omp for private(i, j, left, up, right, down) 
		for (int rp = 0; rp < Np; rp++) {
			//get the (i,j) index of the current point
			int point = red[rp]; 
			index2ij(point, N, i, j);

			//set left, right, up, and down values
			setDirections(i, j, N, guess, left, right, up, down);

			//apply the update formula
			guess[point] = 1.0/4.0*(H*f[point] + up + down + left + right);
		}
		#pragma omp barrier
		#pragma omp for private(i, j, left, up, right, down) 
		//loop over black points
		for (int bp = 0; bp < Nm; bp++) {
			//get the (i,j) index of the current point
			int point = black[bp];
			index2ij(point, N, i, j);

			//set left, right, up, and down values
			setDirections(i, j, N, guess, left, right, up, down);

			//apply the update formula
			guess[point] = 1.0/4.0*(H*f[point] + up + down + left + right);
		}
		}
	}

	//free black and red
	free(black); free(red); 
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

		//apply the matrix A
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

	//initialize initial guess to zeros and rhs to ones
	for (int i = 0; i < N*N; i++) u[i] = 0;
	for (int i = 0; i < N*N; i++) f[i] = 1;

	//apply the iterations, time
	Timer t;
  t.tic();
  gsP(N, f, max_iter, u);
  double time = t.toc();
  printf("Time taken: %3f seconds\n", time);

  //compute a residual as a check
  double r = computeRes(N, u, f);
  printf("Residual : %3f\n", r);

  //free the memory
  free(f); free(u);

  return 0;
}
