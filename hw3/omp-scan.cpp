#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void getChunk(long n, int nt, int tID, int& chunk, int& first, int& last) {
  //get chunk size and first and last entry of thread nt

  //get size of chunk per thread, with possibility of remainder
  int quot = n / nt; int rem = n & nt;

  //assign start and stop locations and chunk size. Remainder goes to final thread
  if (tID == 0) {
    chunk = quot; first = 1; last = (tID+1)*quot;
  }
  else if (tID == nt - 1) {
    chunk = quot + rem; first = tID * quot; last = n;
  }
  else {
    chunk = quot; first = tID * quot; last = (tID+1)*quot;
  }
}

void fixSum(long n, int nt, int* F, long* prefix_sum) {
  //fix the sum at breakpoints stored in F.
  int last = 0; int cum_sum = 0; int j;

  //do the first nt-1 chunks
  for (int i = 0; i < nt - 1; i++) {
    for (j = F[i]; j < F[i+1]-1; j++) {
      prefix_sum[j] += cum_sum;
    }
    last = prefix_sum[j]; prefix_sum[j] += cum_sum;
    cum_sum += last;
  }

  //handle the final chunk
  for (int j = F[nt-1]; j < n; j++) {
    prefix_sum[j] += cum_sum;
  }
}


void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan

  //return if no vector
  if (n == 0) return;

  //initialize prefix sum - shared
  prefix_sum[0] = 0; int* F; int num_threads;

  #pragma omp parallel shared(F, num_threads)
  {
  //set number of threads
  num_threads = omp_get_num_threads();

  //keep track of index of firsts - only one htread makes this
  #pragma omp single
  {
  F = new int[num_threads]; F[0]=1;
  }
  #pragma omp barrier

  //open parallel region
  #pragma omp for 
  for (int i = 0; i < num_threads; i++) {
    //get chunks
    int tID = omp_get_thread_num();
    int chunk = 0, first = 0, last = 0;
    getChunk(n, num_threads, tID, chunk, first, last);
    F[tID] = first;

    //update prefix_sum
    prefix_sum[first] = A[first-1];
    for (int j = first+1; j < last; j++) {
      prefix_sum[j] = prefix_sum[j-1] + A[j-1];
    }
  }
  }

  //fix by adding constants
  fixSum(n, num_threads, F, prefix_sum);

  //free memory
  delete []F;
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand() % 5; 

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
