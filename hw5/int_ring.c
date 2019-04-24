#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

double time_cycle(long N, long Nsize, MPI_Comm comm) {
	//declare ranks for mpi
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size( comm, &size );


  //define the integer to be passed around on process 0
  int* num = (int*) malloc(Nsize*sizeof(int));
    if (rank == 0) for (int i = 0; i < Nsize; i++) num[i] = 0;

  //define number of processes to use
  int num_proc = size;

	//call barrier so timing is accurate
  MPI_Barrier(comm);
  double tt = MPI_Wtime();

  //repeat N times and time
  for (long repeat  = 0; repeat < N; repeat++) {
    MPI_Status status;

    //pass around the data
    if (rank == 0) { //at 0, send to 1, then receieve from the end
    	MPI_Send(num, Nsize, MPI_INT, rank+1, repeat, comm);
    	MPI_Recv(num, Nsize, MPI_INT, num_proc-1, repeat, comm, &status);
    }
    else { //now, recieve, modify, then send. check for last proc
    	MPI_Recv(num, Nsize, MPI_INT, rank-1, repeat, comm, &status);
    	num[0] += rank; 
    	if (rank < num_proc - 1) {
    		MPI_Send(num, Nsize, MPI_INT, rank+1, repeat, comm);
    	}
    	else {
    		MPI_Send(num, Nsize, MPI_INT, 0, repeat, comm);
    	}
    }
  }
  //time the loops
  MPI_Barrier(comm);
  tt = MPI_Wtime() - tt;

  //get the error
  int soln = N*num_proc*(num_proc-1)/2;
  if (rank == 0 && Nsize == 1) printf("Error in sum: %d\n", soln - num[0]);

  return tt;
}

int main(int argc, char** argv) {

	//initialize mpi
  MPI_Init(&argc, &argv);

  //get input 
  if (argc < 2) {
    printf("Usage: mpirun ./int_ring <N> \n");
    return 1;
  }
  int N = atoi(argv[1]);

  //declare rank and communication
  int rank, size;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size );

  //check if size is greater than 1
  if (size == 1) {
    printf("Cannot exhange data with only one process. Exiting...\n");
    return 0;
  }
  else {
    if (rank == 0) printf("Using %d processes.\n", size);
  }

  //do timing for passing single integer
  double tt = time_cycle(N, 1, comm);
  if (!rank) printf("cycle latency: %e ms\n", tt/N * 1000);

  //do timing for passing an array
  long Nsize = 1000000;
  tt = time_cycle(N, Nsize, comm);
  if (!rank) printf("cycle bandwidth: %e GB/s\n", (Nsize*N)/tt/1e9);

  //finalize mpi
  MPI_Finalize();
  return 0;
}