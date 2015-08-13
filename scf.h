// GSL eigen solver for real symmetric matrix
void my_eigen_symmv(gsl_matrix* data, int DIM,
	                gsl_vector* eval, gsl_matrix* evec);

// print gsl matrix
void my_print_matrix(gsl_matrix* A);

// read nuclear repulsion energy
double get_nucl_ene(void);
 
// get number of basis functions
int get_nbasis(void);

// read overlap integral
void read_overlap(int nbasis, gsl_matrix *S);

// read kinetic energy integral
void read_kinetic(int nbasis, gsl_matrix *T);

// read nuclear attraction integral
void read_potential(int nbasis, gsl_matrix *V);

// read two-electron integrals
void read_eri(int n_eri, double *ERI);

// get core Hamiltonian
void sum_H_core(int nbasis, gsl_matrix *H_core, gsl_matrix *T, gsl_matrix *V);

// diagonalize overlap matrix
void diag_overlap(int nbasis, gsl_matrix *S, gsl_matrix *S_invsqrt);

// from Fock matrix to MO coeffcients
void Fock_to_Coef(int nbasis, gsl_matrix *Fock, gsl_matrix *S_invsqrt, 
				  gsl_matrix *Coef, gsl_vector *emo);

// from MO coeffcients to density matrix
void Coef_to_Dens(int nbasis, int n_occ, gsl_matrix *Coef, gsl_matrix *D);

// compute the initial SCF energy
double get_elec_ene(int nbasis, gsl_matrix *D, gsl_matrix *H_core, 
					gsl_matrix *Fock);

// form G matrix
void form_G(int nbasis, gsl_matrix *D_prev, double *ERI, gsl_matrix *G);

// form Fock matrix
void form_Fock(int nbasis, gsl_matrix *H_core, gsl_matrix *G, gsl_matrix *Fock);
