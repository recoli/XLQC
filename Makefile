CC=g++
CFLAGS=-std=c++0x -O2 -Wall -lgsl -lgslcblas
OBJ=basis.o scf.o main.o int_lib/crys.o int_lib/chgp.o int_lib/cints.o

%.o: %.cc
	$(CC) -c -o $@ $< $(CFLAGS)

main: $(OBJ)
	g++ -o $@ $^ $(CFLAGS)

clean:  
	rm -f main *.o int_lib/*.o
