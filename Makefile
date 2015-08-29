CC=g++
CFLAGS=-std=c++0x -O2 -Wall -lgsl -lgslcblas

NVCC=nvcc
NVCFLAGS=-lgsl -lgslcblas

OBJ=basis.o scf.o main.o cuda_rys_sp.o int_lib/crys.o int_lib/cints.o

main: $(OBJ)
	$(NVCC) -o $@ $^ $(NVCFLAGS)

%.o: %.cu
	$(NVCC) -dc -o $@ $< $(NVCFLAGS)

%.o: %.cc
	$(CC) -c -o $@ $< $(CFLAGS)

clean:  
	rm -f main *.o int_lib/*.o
