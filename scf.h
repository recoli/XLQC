// copy the content of a matrix
void mat_mem_cpy(int DIM, double **A, double **B);

// matrix inner product
double mat_inn_prod(int DIM, double **A, double **B);

// GSL eigen solver for real symmetric matrix
void my_eigen_symmv(gsl_matrix* data, int DIM,
	                gsl_vector* eval, gsl_matrix* evec);

// print gsl matrix
void my_print_matrix(gsl_matrix* A);

// print gsl vector
void my_print_vector(gsl_vector* x);

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

// DIIS
void update_Fock_DIIS(int *p_diis_dim, int *p_diis_index, double *p_delta_DIIS, 
						gsl_matrix *Fock, gsl_matrix *D_prev, gsl_matrix *S, Basis *p_basis,
						double ***diis_err, double ***diis_Fock);
