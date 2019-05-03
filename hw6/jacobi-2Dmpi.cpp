#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

void index2ij(int index, int length, int& i, int& j) {
	//maps an index of an array to an i and j value
	i = index / length;
	j = index % length;
}

int toIndex(int i, int j, int N) {
	//maps (i,j) to column based index
	return j+N*i;
}

void setDirections(int i, int j, int N, double* vec, double& left, double& right,
																											double& up, double& down) {
	//set the values of u at left, right, up, and down of current point (i,j)
	//handle the case where point is at bottom row
	up = vec[toIndex(i+1, j, N)];
	right = vec[toIndex(i, j+1, N)];
	down = vec[toIndex(i-1, j, N)];
	left = vec[toIndex(i, j-1, N)];
}

void getTop(int Nl, int proc, int i, int j, int proc_per_row, double* lu, double* buffer) {
	//get the top row of ghost cells for this processor

	MPI_Status status;

	//check if procesor is NOT at top of array
	//sending
	if (i > 0) {//this proc is sending data down
		int sender = proc;
		int receiver = toIndex(i-1, j, proc_per_row); //proc rank of receiver
		//fill the buffer
		for (int q = 0; q < Nl+2; q++) {
			buffer[q] = lu[Nl+2+q];
		}
		//send the buffer
		//printf("proc %d sending to proc %d\n", sender, receiver);
		MPI_Send(buffer, Nl+2, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
	}
	//receiving
	if (i < proc_per_row - 1) {//this proc is receiving data
		int sender = toIndex(i+1, j, proc_per_row); //proc rank of sender
		int receiver = proc;
		//printf("proc %d set to receive from %d\n", receiver, sender);
		MPI_Recv(buffer, Nl+2, MPI_DOUBLE, sender, 0, MPI_COMM_WORLD, &status);
		//transfer from buffer to array
		for (int q = 0; q < Nl+2; q++) {
			lu[(Nl+1)*(Nl+2)+q] = buffer[q];
		}
	}

}

void getBottom(int Nl, int proc, int i, int j, int proc_per_row, double* lu, double* buffer) {
	//get the top row of ghost cells for this processor

	MPI_Status status;

	//check if procesor is NOT at bottom of array
	//sending
	if (i < proc_per_row -1) {//this proc is sending data up
		int sender = proc;
		int receiver = toIndex(i+1, j, proc_per_row); //proc rank of receiver
		//fill the buffer
		for (int q = 0; q < Nl+2; q++) {
			buffer[q] = lu[(Nl)*(Nl+2)+q];
		}
		//send the buffer
		//printf("proc %d sending to proc %d\n", sender, receiver);
		MPI_Send(buffer, Nl+2, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
	}
	//receiving
	if (i > 0) {//this proc is receiving data
		int sender = toIndex(i-1, j, proc_per_row); //proc rank of sender
		int receiver = proc;
		//printf("proc %d set to receive from %d\n", receiver, sender);
		MPI_Recv(buffer, Nl+2, MPI_DOUBLE, sender, 0, MPI_COMM_WORLD, &status);
		//transfer from buffer to array
		for (int q = 0; q < Nl+2; q++) {
			lu[q] = buffer[q];
		}
	}

}

void getLeft(int Nl, int proc, int i, int j, int proc_per_row, double* lu, double* buffer) {
	//get the top row of ghost cells for this processor

	MPI_Status status;

	//check if procesor is NOT at left of array
	//sending
	if (j < proc_per_row -1) {//this proc is sending data right
		int sender = proc;
		int receiver = toIndex(i, j+1, proc_per_row); //proc rank of receiver
		//fill the buffer
		for (int q = 0; q < Nl+2; q++) {
			buffer[q] = lu[(q+1)*(Nl+2)-2];
		}
		//send the buffer
		//printf("proc %d sending to proc %d\n", sender, receiver);
		MPI_Send(buffer, Nl+2, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
	}
	//receiving
	if (j > 0) {//this proc is receiving data
		int sender = toIndex(i, j-1, proc_per_row); //proc rank of sender
		int receiver = proc;
		//printf("proc %d set to receive from %d\n", receiver, sender);
		MPI_Recv(buffer, Nl+2, MPI_DOUBLE, sender, 0, MPI_COMM_WORLD, &status);
		//transfer from buffer to array
		for (int q = 0; q < Nl+2; q++) {
			lu[(q)*(Nl+2)] = buffer[q];
		}
	}

}

void getRight(int Nl, int proc, int i, int j, int proc_per_row, double* lu, double* buffer) {
	//get the top row of ghost cells for this processor

	MPI_Status status;

	//check if procesor is NOT at right of array
	//sending
	if (j > 0) {//this proc is sending data left
		int sender = proc;
		int receiver = toIndex(i, j-1, proc_per_row); //proc rank of receiver
		//fill the buffer
		for (int q = 0; q < Nl+2; q++) {
			buffer[q] = lu[q*(Nl+2)+1];
		}
		//send the buffer
		//printf("proc %d sending to proc %d\n", sender, receiver);
		MPI_Send(buffer, Nl+2, MPI_DOUBLE, receiver, 0, MPI_COMM_WORLD);
	}
	//receiving
	if (j < proc_per_row - 1) {//this proc is receiving data
		int sender = toIndex(i, j+1, proc_per_row); //proc rank of sender
		int receiver = proc;
		//printf("proc %d set to receive from %d\n", receiver, sender);
		MPI_Recv(buffer, Nl+2, MPI_DOUBLE, sender, 0, MPI_COMM_WORLD, &status);
		//transfer from buffer to array
		for (int q = 0; q < Nl+2; q++) {
			lu[(q+1)*(Nl+2)-1] = buffer[q];
		}
		if (sender == 1 && receiver == 0) {
			//printf("ex %f\n", lu[7]);
			//for (int q = 0; q < Nl+2; q++) printf("buff %f\n", buffer[q]);
			//for (int q = 0; q < (Nl+2)*(Nl+2); q++) printf("lu %f\n", lu[q]);
		}
	}

}

void commGhost(int Nl, int mpirank, int proc_per_row, double* lu, double* buffer) {
	//communicate the data in ghost nodes with mpi
	int i, j; //indices in the processor array
	index2ij(mpirank, proc_per_row, i, j); //get indices
	getTop(Nl, mpirank, i, j, proc_per_row, lu, buffer);
	getBottom(Nl, mpirank, i, j, proc_per_row, lu, buffer);
	getLeft(Nl, mpirank, i, j, proc_per_row, lu, buffer);
	getRight(Nl, mpirank, i, j, proc_per_row, lu, buffer);
}


double compute_residual(double* lu, int Nl, double invhsq) {
	//compute global residual, assuming ghost cells updated
  int i,j; double left,right,up,down;
  double tmp, gres = 0.0, lres = 0.0;

  for (int index = 0; index < (Nl+2)*(Nl+2); index++){
  	index2ij(index, Nl+2, i, j);
		if ( i > 0 && i < Nl+1 && j > 0 && j < Nl+1) {//if its an inside pt, compute
			setDirections(i, j, Nl+2, lu, left, right, up, down);
    	tmp = ((4.0*lu[i] - left - right - up - down) * invhsq - 1);
    	//printf("%f\n",tmp);
    	lres += tmp * tmp;
    }
  }
  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}


void jacobi2Dmpi(int N, int Nl, int max_iters, double* lu, double* lf) {
	//perform 2d jacobi with mpi blocking

	//get processor info
	int p, mpirank;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  int proc_per_row = sqrt(p+0.001); //processors per row in array

  //grid and iteration parameters
  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1.0/hsq;
	double gres, gres0, tol = 1e-5;
	int i,j,index; //array indices
	double left, up, right, down; //for jacobi update

	//create new grid to compute update into
	double* lunew = new double[(Nl+2)*(Nl+2)];
	for (int k = 0; k < (Nl+2)*(Nl+2); k++) lunew[k] = 0;
	double* buffer = new double[Nl+2];


	/* initial residual */
  gres0 = compute_residual(lu, Nl, invhsq);
  gres = gres0;

  //do iteration
	for (int iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {
		//start with a local update, assuming ghost points are updated
		for (index = 0; index < (Nl+2)*(Nl+2); index++) {//loop over every local grid pt
			index2ij(index, Nl+2, i, j);
			if ( i > 0 && i < Nl+1 && j > 0 && j < Nl+1) {//if its an inside pt, update
				setDirections(i, j, Nl+2, lu, left, right, up, down);
				lunew[index] = 1.0/4.0 * (hsq*lf[index] + up + down + left + right);
				//if (index == 6 && mpirank == 0) printf("%f,%f,%f,%f,%f\n", left,down,right,up,lunew[index]);
				//printf("%f\n", lunew[index]);
			}
		}

		//now we need to communicate ghost nodes
		commGhost(Nl, mpirank, proc_per_row, lunew, buffer);

		//copy new values into the old array
		for (index = 0; index < (Nl+2)*(Nl+2); index++) {
			lu[index]=lunew[index];
		}

		//get new residual
		/*
		if (0 == (iter % 10)) {
      gres = compute_residual(lu, Nl, invhsq);
      if (0 == mpirank) {
				printf("Iter %d: Residual: %g\n", iter, gres);
      }
    }
    */
	}

	//iterations done
	delete []lunew; delete []buffer;
}



int main(int argc, char * argv[]){
	//declare variables
  int mpirank, p, N, Nl, max_iters;
  MPI_Status status;

  //check if input is correct
  if (argc != 3) {
		fprintf(stderr, "Usage: <N> <max_iters> %s\n", argv[0]);
		return 1;
	}

	//store input
  sscanf(argv[1], "%d", &N);
	sscanf(argv[2], "%d", &max_iters);

	//initialize mpi
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  //check if the dimensions given are consistent. p=4^j. N=2^j*Nl
  int j = log(p)/log(4); 
  if (pow(4,j) != p) {
  	fprintf(stderr, "Number of processors not divisible by 4^j.\n");
  	MPI_Abort(MPI_COMM_WORLD, 0);
  }
  Nl = N / pow(2,j);
  if (pow(2,j) * Nl != N) {
  	fprintf(stderr, "N is not divisible by 2^j.\n");
  	MPI_Abort(MPI_COMM_WORLD, 0);
  }

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

  //parameters work out, make a local grid, initialize to 0. initialize f to 1
  double* lu = new double[(Nl+2)*(Nl+2)];
  for (int i = 0; i < (Nl+2)*(Nl+2); i++) lu[i] = 0;
  double* lf = new double[(Nl+2)*(Nl+2)];
  for (int i = 0; i < (Nl+2)*(Nl+2); i++) lf[i] = 1;
  

	//call jacobi method
  jacobi2Dmpi(N, Nl, max_iters, lu, lf);


	//if (mpirank == 0)printf("Ex value: %f, %f, %f, %f\n", lu[5], lu[6], lu[9], lu[10]);

	//free memory and finalize mpi
  delete []lu; 
  MPI_Finalize();
	return 0;
}