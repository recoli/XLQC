CC=gcc
CFLAGS=-O3 -Wall -lgsl -lgslcblas
OBJ=basis.o scf.o main.o int_lib/crys.o int_lib/chgp.o int_lib/cints.o

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

main: $(OBJ)
	gcc -o $@ $^ $(CFLAGS)

clean:  
	rm -f main *.o int_lib/*.o
