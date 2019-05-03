CC = mpicc # or mpicxx
CXX = mpic++
CFLAGS = -O3 -march=native -fopenmp -std=c99
CXXFLAGS = -O3 -march=native -fopenmp -std=c++11

TARGETS = $(basename $(wildcard *.c)) $(basename $(wildcard *.cpp))

all : $(TARGETS)

%:%.c *.h
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

%:%.cpp *.h
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@
clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
