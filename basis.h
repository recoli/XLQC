// allocate memory with failure checking
void* my_malloc(size_t bytes);

void* my_malloc_2(size_t bytes, char *word);

// combination index
int ij2intindex(int i, int j);

// get nuclear charge of an element
int get_nuc_chg(char *element);

// get number of atoms 
int get_natoms(void);

// read geometry
void read_geom(Atom *p_atom);

double calc_ene_nucl(Atom *p_atom);

// parse basis set
// get number of basis functions
void parse_basis(Atom *p_atom, Basis *p_basis);

// read the full basis set created by parse_basis
void read_basis(Atom *p_atom, Basis *p_basis);

void print_basis(Basis *p_basis);

double calc_int_overlap(Basis *p_basis, int a, int b);
double calc_int_kinetic(Basis *p_basis, int a, int b);
double calc_int_nuc_attr(Basis *p_basis, int a, int b, Atom *p_atom);
double calc_int_eri_hgp(Basis *p_basis, int a, int b, int c, int d);
