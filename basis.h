// allocate memory with failure checking
void* my_malloc(size_t bytes);

// combination index
int ij2intindex(int i, int j);

// get nuclear charge of an element
int get_nuc_chg(char *element);

// get number of atoms 
int get_natoms(void);

// read geometry
void read_geom(double **atom_pos, int *atom_nuc_chg, char **atom_name);

// parse basis set
// get number of basis functions
void parse_basis(int natoms, char **atom_name, int *atom_nuc_chg, int *p_nbasis);

// read the full basis set created by parse_basis
void read_basis(int natoms, double **atom_pos, int nbasis, double **expon, double **coef, 
				int *nprims, int **lmn, double **xbas);
