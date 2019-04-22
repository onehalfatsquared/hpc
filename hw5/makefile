CC = mpicc # or mpicxx
CFLAGS = -O3 -march=native -fopenmp

TARGETS = $(basename $(wildcard *.c))

all : $(TARGETS)

%:%.c *.h
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean