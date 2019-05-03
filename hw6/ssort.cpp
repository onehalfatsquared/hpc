// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>




void sample(int L, int* big, int l, int* small) {
  //samples sample every L/(l+1) element of big, put in small
  for (int  i = 0; i < l; i++) {
    small[i] = big[(i+1)*L/(l+1)];
  }
}

int main( int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the command line
  int N;

  //check if input is correct
  if (argc != 2) {
    fprintf(stderr, "Usage: <N> %s\n", argv[0]);
    return 1;
  }
  
  //store input
  sscanf(argv[1], "%d", &N);

  int* vec = (int*)malloc(N*sizeof(int));
  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  printf("rank: %d, first entry: %d\n", rank, vec[0]);

  // sort locally
  std::sort(vec, vec+N);

  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector
  int* splitters = (int*) malloc((p-1)*sizeof(int));
  sample(N, vec, p-1, splitters);

  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
  int* allSplitter = NULL;
  if (rank == 0) {
    allSplitter = (int* ) malloc(p*(p-1)*sizeof(int));
  }
  MPI_Gather(splitters, p-1, MPI_INT, allSplitter, p-1, MPI_INT, 0, MPI_COMM_WORLD);

  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
  if (rank == 0) {
    std::sort(allSplitter, allSplitter + p*(p-1));
    sample(p*(p-1), allSplitter, p-1, splitters);
  }

  // root process broadcasts splitters to all other processes
  MPI_Bcast(splitters, p-1, MPI_INT, 0, MPI_COMM_WORLD);

  // every process uses the obtained splitters to decide which
  // integers need to be sent to which other process (local bins).
  // Note that the vector is already locally sorted and so are the
  // splitters; therefore, we can use std::lower_bound function to
  // determine the bins efficiently.
  //
  // Hint: the MPI_Alltoallv exchange in the next step requires
  // send-counts and send-displacements to each process. Determining the
  // bins for an already sorted array just means to determine these
  // counts and displacements. For a splitter s[i], the corresponding
  // send-displacement for the message to process (i+1) is then given by,
  // sdispls[i+1] = std::lower_bound(vec, vec+N, s[i]) - vec;
  int* sdispls = (int*) malloc(p*sizeof(int)); sdispls[0] = 0;
  for (int i = 1; i < p; i++) {
    sdispls[i] = std::lower_bound(vec, vec+N, splitters[i-1])-vec;
  }
  //if (rank == test) for (int i = 0; i < p; i++) printf("%d\n",sdispls[i]);
  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect.
  int* sendcounts = (int*) malloc(p*sizeof(int));
  for (int i = 0; i < p-1; i++) {
    sendcounts[i] = sdispls[i+1]-sdispls[i];
  }
  sendcounts[p-1] = N - sdispls[p-1];
  //make array that stores how much data is coming from each proc
  int* incoming = (int*) malloc(p*sizeof(int)); 
  MPI_Alltoall(sendcounts, 1, MPI_INT, incoming, 1, MPI_INT, MPI_COMM_WORLD);
  //make array that will store final values - size is sum of incoming
  int S = 0; for(int i = 0; i < p; i++) S += incoming[i];
  int* final = (int*) malloc(S*sizeof(int)); 
  //make array with receive displacements
  int* rdispls = (int*) malloc(p*sizeof(int));
  int S2 = 0;
  for (int i = 0; i < p; i++) {
    rdispls[i] = S2;
    S2 += incoming[i];
  }

  //and then use MPI_Alltoallv to exchange the data
  MPI_Alltoallv(vec, sendcounts, sdispls, MPI_INT, final, incoming, rdispls, MPI_INT, MPI_COMM_WORLD);

  // do a local sort of the received data
  std::sort(final, final + S);

  // every process writes its result to a file
  FILE* fd = NULL;
  char filename[256];
  snprintf(filename, 256, "output%02d.txt", rank);
  fd = fopen(filename,"w+");

  if(NULL == fd) {
    printf("Error opening file \n");
    return 1;
  }

  for(int i = 0; i < S; i++)
    fprintf(fd, "%d\n", final[i]);
  fclose(fd);

  //free memory and finalize
  free(vec); free(splitters); free(sdispls); free(sendcounts);
  free(incoming); free(rdispls); free(final);
  if (rank == 0) {
    free(allSplitter);
  }
  MPI_Finalize();
  return 0;
}