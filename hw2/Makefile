EXEC = MMult1 val_test01_solved val_test02_solved omp_solved2 omp_solved3 \
	omp_solved4 omp_solved5 omp_solved6 jacobi2D-omp gs2D-omp
CC = g++
FLAGS = -fopenmp -O3

all: $(EXEC)

MMutl1: MMult1.cpp
	$(CC) $(FLAGS) MMult1.cpp -o MMult1

val_test01_solved: val_test01_solved.cpp
	$(CC) $(FLAGS) val_test01_solved.cpp -o val_test01_solved

val_test02_solved: val_test02_solved.cpp
	$(CC) $(FLAGS) val_test02_solved.cpp -o val_test02_solved

omp_solved2: omp_solved2.c
	$(CC) $(FLAGS) omp_solved2.c -o omp_solved2

omp_solved3: omp_solved3.c
	$(CC) $(FLAGS) omp_solved3.c -o omp_solved3

omp_solved4: omp_solved4.c
	$(CC) $(FLAGS) omp_solved4.c -o omp_solved4

omp_solved5: omp_solved5.c
	$(CC) $(FLAGS) omp_solved5.c -o omp_solved5

omp_solved6: omp_solved6.c
	$(CC) $(FLAGS) omp_solved6.c -o omp_solved6

jacobi2D-omp: jacobi2D-omp.cpp
	$(CC) $(FLAGS) -o jacobi2D-omp jacobi2D-omp.cpp 

gs2D-omp: gs2D-omp.cpp
	$(CC) $(FLAGS) -o gs2D-omp gs2D-omp.cpp 

clean:
	rm -rf $(EXEC) *.out

