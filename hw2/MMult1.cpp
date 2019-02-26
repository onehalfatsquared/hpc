// g++ -fopenmp -O3 -march=native MMult1.cpp && ./a.out

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

#define BLOCK_SIZE 32

// Note: matrices are stored in column major order; i.e. the array elements in
// the (m x n) matrix C are stored in the sequence: {C_00, C_10, ..., C_m0,
// C_01, C_11, ..., C_m1, C_02, ..., C_0n, C_1n, ..., C_mn}
void MMult0(long m, long n, long k, double *a, double *b, double *c) {
  //does column first operatoins 
  for (long j = 0; j < n; j++) {
    for (long p = 0; p < k; p++) {
      for (long i = 0; i < m; i++) {
        double A_ip = a[i+p*m];
        double B_pj = b[p+j*k];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }
}

void MMult1(long m, long n, long k, double *a, double *b, double *c) {
  //does row first operations
  for (long i = 0; i < m; i++) {
    for (long p = 0; p < k; p++) {
      for (long j = 0; j < n; j++) {
        double A_ip = a[i+p*m];
        double B_pj = b[p+j*k];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }
}

int toIndex(int r, int c, long m) {
  //map row and column number into index in 1d array. column indexed
  return m*c+r;
}

void readBlock(int i, int j, long m, double *M, double* B) {
  //read in the (i,j) block of a matrix M, store in B
  for (int x = 0; x < BLOCK_SIZE; x++) {
    for (int y = 0; y < BLOCK_SIZE; y++) {
      int row = BLOCK_SIZE*i + y;
      int column = BLOCK_SIZE*j + x;
      B[toIndex(y, x, BLOCK_SIZE)] = M[toIndex(row, column, m)];
    }
  }
}

void writeBlock(int i, int j, long m, double* source, double* target) {
  //writes block i,j source back into the full matrix target
  for (int x = 0; x < BLOCK_SIZE; x++) {
    for (int y = 0; y < BLOCK_SIZE; y++) {
      int row = BLOCK_SIZE*i + y;
      int column = BLOCK_SIZE*j + x;
      target[toIndex(row, column, m)] = source[toIndex(y, x, BLOCK_SIZE)];
    }
  }
}

void MMultBlock(long m, int num_blocks, double *a, double *b, double *c) {
  // implements a column first blocking procedure
  double* A_ip = (double*) aligned_malloc( BLOCK_SIZE*BLOCK_SIZE * sizeof(double));//initialize block storage
  double* B_pj = (double*) aligned_malloc( BLOCK_SIZE*BLOCK_SIZE * sizeof(double));//initialize block storage
  double* C_ij = (double*) aligned_malloc( BLOCK_SIZE*BLOCK_SIZE * sizeof(double));//initialize block storage
  for (long i = 0; i < BLOCK_SIZE*BLOCK_SIZE; i++) A_ip[i]=B_pj[i]=C_ij[i] = 0;
  for (long i = 0; i < num_blocks; i++) {
    for (long j = 0; j < num_blocks; j++) {
      readBlock(i, j, m, c, C_ij); //read in block Cij
      for (long p = 0; p < num_blocks; p++) {
        readBlock(i, p, m, a, A_ip); //read in block Aip
        readBlock(p, j, m, b, B_pj); //read in block Bpj
        MMult0(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, A_ip, B_pj, C_ij);   //multiply and add blocks
        writeBlock(i, j, m, C_ij, c);     //write the block back to storage
      }
    }
  }
  //free the temp memory
  aligned_free(A_ip); aligned_free(B_pj); aligned_free(C_ij); 
}

void toBlock(int block_num, int num_blocks, int& i, int& j) {
  // map a column index 1-d block array into (i,j)
  i = block_num % num_blocks;
  j = block_num / num_blocks;
}

void MMultBlockP(long m, int num_blocks, double *a, double *b, double *c) {
  // implements a column first blocking procedure - parallelized
  #pragma omp parallel
  {
    //allocate temp memory for blocks
    double* A_ip = (double*) aligned_malloc( BLOCK_SIZE*BLOCK_SIZE * sizeof(double));//initialize block storage
    double* B_pj = (double*) aligned_malloc( BLOCK_SIZE*BLOCK_SIZE * sizeof(double));//initialize block storage
    double* C_ij = (double*) aligned_malloc( BLOCK_SIZE*BLOCK_SIZE * sizeof(double));//initialize block storage
    for (long i = 0; i < BLOCK_SIZE*BLOCK_SIZE; i++) A_ip[i]=B_pj[i]=C_ij[i] = 0;
    #pragma omp  for
    for (int block = 0; block < num_blocks*num_blocks; block++) {
      int i,j; 
      toBlock(block, num_blocks, i, j); 
      readBlock(i, j, m, c, C_ij); //read in block Cij
      for (long p = 0; p < num_blocks; p++) {
        readBlock(i, p, m, a, A_ip); //read in block Aip
        readBlock(p, j, m, b, B_pj); //read in block Bpj
        MMult0(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, A_ip, B_pj, C_ij);   //multiply and add blocks
        writeBlock(i, j, m, C_ij, c);     //write the block back to storage
      }
    }
    //free the temp memory
    aligned_free(A_ip); aligned_free(B_pj); aligned_free(C_ij); 
  }
}

int main(int argc, char** argv) {
  const long PFIRST = BLOCK_SIZE;
  const long PLAST = 2000;
  const long PINC = std::max(50/BLOCK_SIZE,1) * BLOCK_SIZE; // multiple of BLOCK_SIZE

  printf(" Dimension       Time    Gflop/s       GB/s        Error\n");
  for (long p = PFIRST; p < PLAST; p += PINC) {
    long m = p, n = p, k = p;
    long NREPEATS = 1e9/(m*n*k)+1;
    double* a = (double*) aligned_malloc(m * k * sizeof(double)); // m x k
    double* b = (double*) aligned_malloc(k * n * sizeof(double)); // k x n
    double* c = (double*) aligned_malloc(m * n * sizeof(double)); // m x n
    double* c_ref = (double*) aligned_malloc(m * n * sizeof(double)); // m x n
    

    // Initialize matrices
    for (long i = 0; i < m*k; i++) a[i] = drand48(); 
    for (long i = 0; i < k*n; i++) b[i] = drand48();
    for (long i = 0; i < m*n; i++) c_ref[i] = 0;
    for (long i = 0; i < m*n; i++) c[i] = 0;

    //Set number of blocks
    int num_blocks = n / BLOCK_SIZE; //number of block, assumes square matrix

    //run matrix multiplications
    for (long rep = 0; rep < NREPEATS; rep++) { // Compute reference solution
      MMult0(m, n, k, a, b, c_ref);
    }

    //time the runs
    Timer t;
    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) {
      //MMult0(m, n, k, a, b, c);
      //MMult1(m, n, k, a, b, c);
      //MMultBlock(m, num_blocks, a, b, c);
      MMultBlockP(m, num_blocks, a, b, c);
    }
    double time = t.toc();
    double flops = 0; // TODO: calculate from m, n, k, NREPEATS, time
    flops = 2*NREPEATS*m*n*k/(1e9)/time; 
    double bandwidth = 0; // TODO: calculate from m, n, k, NREPEATS, time
    bandwidth = NREPEATS*4*m*n*k*sizeof(double)/1e9/time;
    printf("%10ld %10f %10f %10f", p, time, flops, bandwidth);

    //compute error with reference
    double max_err = 0;
    for (long i = 0; i < m*n; i++) max_err = std::max(max_err, fabs(c[i] - c_ref[i]));
    printf(" %10e\n", max_err);

    //free the memory
    aligned_free(a); aligned_free(b); aligned_free(c);
  }

  return 0;
}

// * Using MMult0 as a reference, implement MMult1 and try to rearrange loops to
// maximize performance. Measure performance for different loop arrangements and
// try to reason why you get the best performance for a particular order?
//
//
// * You will notice that the performance degrades for larger matrix sizes that
// do not fit in the cache. To improve the performance for larger matrices,
// implement a one level blocking scheme by using BLOCK_SIZE macro as the block
// size. By partitioning big matrices into smaller blocks that fit in the cache
// and multiplying these blocks together at a time, we can reduce the number of
// accesses to main memory. This resolves the main memory bandwidth bottleneck
// for large matrices and improves performance.
//
// NOTE: You can assume that the matrix dimensions are multiples of BLOCK_SIZE.
//
//
// * Experiment with different values for BLOCK_SIZE (use multiples of 4) and
// measure performance.  What is the optimal value for BLOCK_SIZE?
//
//
// * Now parallelize your matrix-matrix multiplication code using OpenMP.
//
//
// * What percentage of the peak FLOP-rate do you achieve with your code?
//
//
// NOTE: Compile your code using the flag -march=native. This tells the compiler
// to generate the best output using the instruction set supported by your CPU
// architecture. Also, try using either of -O2 or -O3 optimization level flags.
// Be aware that -O2 can sometimes generate better output than using -O3 for
// programmer optimized code.
