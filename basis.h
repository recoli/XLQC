// allocate memory with failure checking
void* my_malloc(size_t bytes);

// combination index
int ij2intindex(int i, int j);

// get nuclear charge of an element
int get_nuc_chg(char *element);

// get number of atoms 
int get_natoms(void);

// read geometry
void read_geom(Atom *p_atom);

// parse basis set
// get number of basis functions
void parse_basis(Atom *p_atom, Basis *p_basis);

// read the full basis set created by parse_basis
void read_basis(Atom *p_atom, Basis *p_basis);
