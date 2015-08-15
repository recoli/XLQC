#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

#include "typedef.h"
#include "int_lib/cints.h"

//===============================================
// copy the content of a matrix
//===============================================
void mat_mem_cpy(int DIM, double **A, double **B)
{
	int row, col;
	for (row = 0; row < DIM; ++ row)
	{
		for (col = 0; col < DIM; ++ col)
		{
			A[row][col] = B[row][col];
		}
	}
}

//===============================================
// matrix inner product
//===============================================
double mat_inn_prod(int DIM, double **A, double **B)
{
	double inn_prod = 0.0;
	int row, col;
	for (row = 0; row < DIM; ++ row)
	{
		for (col = 0; col < DIM; ++ col)
		{
			inn_prod += A[row][col] * B[row][col];
		}
	}
	return inn_prod;
}

//===============================================
// GSL eigen solver for real symmetric matrix
//===============================================
void my_eigen_symmv(gsl_matrix* data, int DIM,
	                gsl_vector* eval, gsl_matrix* evec)
{
	if ( DIM <= 0 || data[0].size1 != DIM || data[0].size2 != DIM )
	{
	   printf ("Error: incorrect DIM in my_eigen_symmv!\n");
	   exit(1);
	}

	// make a copy of 'data': 'data_cp' 
	// NOTE: 'data_cp' will be destroyed after gsl_eigen_symmv 
	gsl_matrix *data_cp = gsl_matrix_alloc(DIM, DIM);
	gsl_matrix_memcpy(data_cp, data);

	// diagonalize real symmetric matrix data_cp
	gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc (DIM);
	gsl_eigen_symmv(data_cp, eval, evec, w);
	gsl_eigen_symmv_free(w);

	gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_VAL_ASC);

	gsl_matrix_free(data_cp);
}

//===============================
// print gsl matrix
//===============================
void my_print_matrix(gsl_matrix* A)
{
	int row, col;
	for (row=0; row < A->size1; row++)
	{
		for (col=0; col < A->size2; col++)
		{
			printf("%12.7f", gsl_matrix_get(A, row, col));
		}
		printf("\n");
	}
}

//===============================
// print gsl vector
//===============================
void my_print_vector(gsl_vector* x)
{
	int row;
	for (row=0; row < x->size; row++)
	{
		printf("%12.7f\n", gsl_vector_get(x, row));
	}
}

//===============================
// get core Hamiltonian
//===============================
void sum_H_core(int nbasis, gsl_matrix *H_core, gsl_matrix *T, gsl_matrix *V)
{
	int row, col;
	for (row=0; row<nbasis; row++)
	{
	   for (col=0; col<nbasis; col++)
	   {
			gsl_matrix_set(H_core, row, col, 
					gsl_matrix_get(T, row, col) + gsl_matrix_get(V, row, col));
	   }
	}
}

//===============================
// diagonalize overlap matrix
//===============================
void diag_overlap(int nbasis, gsl_matrix *S, gsl_matrix *S_invsqrt)
{	
	// diagonalization of S
	// eig_S: eigenvalues
	// LS: eigenvectors
	gsl_vector *eig_S = gsl_vector_alloc(nbasis);
	gsl_matrix *LS = gsl_matrix_alloc(nbasis, nbasis);
	my_eigen_symmv(S, nbasis, eig_S, LS);

	// AS: diagonal matrix containing eigenvalues
	// AS_invsqrt: AS^-1/2
	gsl_matrix *AS_invsqrt = gsl_matrix_alloc(nbasis, nbasis);
	gsl_matrix_set_zero(AS_invsqrt);
	int row;
	for (row=0; row < nbasis; row++)
	{
		gsl_matrix_set(AS_invsqrt, row, row, 
				1.0 / sqrt(gsl_vector_get(eig_S, row)));
	}

	// S^-1/2 = LS * AS^-1/2 * LS(T)
	gsl_matrix *prod = gsl_matrix_alloc(nbasis, nbasis);
	gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, LS, AS_invsqrt, 0.0, prod);
	gsl_blas_dgemm (CblasNoTrans, CblasTrans, 1.0, prod, LS, 0.0, S_invsqrt);

	gsl_vector_free (eig_S);
	gsl_matrix_free (LS);
	gsl_matrix_free (AS_invsqrt);
	gsl_matrix_free (prod);
}

//===============================
// from Fock matrix to MO coeffcients
//===============================
void Fock_to_Coef(int nbasis, gsl_matrix *Fock, gsl_matrix *S_invsqrt, 
				  gsl_matrix *Coef, gsl_vector *emo)
{	
	// F' = S^-1/2 * F * S^-1/2
	gsl_matrix *Fock_p = gsl_matrix_alloc(nbasis, nbasis);
	gsl_matrix *prod   = gsl_matrix_alloc(nbasis, nbasis);
	gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, S_invsqrt, Fock, 0.0, prod);
	gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, prod, S_invsqrt, 0.0, Fock_p);


	// diagonalization of Fock_p
	// emo: eigenvalues
	// Coef_p: eigenvectors
	gsl_matrix *Coef_p = gsl_matrix_alloc(nbasis, nbasis);
	my_eigen_symmv(Fock_p, nbasis, emo, Coef_p);

	// C = S^-1/2 * C'
	gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, S_invsqrt, Coef_p, 0.0, Coef);


	gsl_matrix *t1 = gsl_matrix_alloc(nbasis, nbasis);
	gsl_matrix *t2 = gsl_matrix_alloc(nbasis, nbasis);
	gsl_blas_dgemm (CblasTrans, CblasNoTrans, 1.0, Coef_p, Fock_p, 0.0, t1);
	gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, t1, Coef_p, 0.0, t2);


	gsl_matrix_free (prod);
	gsl_matrix_free (Fock_p);
	gsl_matrix_free (Coef_p);
}

//===============================
// from MO coeffcients to density matrix
//===============================
void Coef_to_Dens(int nbasis, int n_occ, gsl_matrix *Coef, gsl_matrix *D)
{	
	int row, col;
	for (row=0; row < nbasis; row++)
	{
	   for (col=0; col < nbasis; col++)
		{
			double val = 0.0;
			int m;
	   		for (m=0; m < n_occ; m++)
			{
				val += gsl_matrix_get(Coef, row, m) * gsl_matrix_get(Coef, col, m);
			}
			gsl_matrix_set(D, row, col, 2.0 * val);
		}
	}
}

//===============================
// compute the initial SCF energy
//===============================
double get_elec_ene(int nbasis, gsl_matrix *D, gsl_matrix *H_core, 
					gsl_matrix *Fock)
{	
	double ene_elec = 0.0;
	int row, col;
	for (row=0; row < nbasis; row++)
	{
	   for (col=0; col < nbasis; col++)
	   {
			ene_elec += 0.5 * gsl_matrix_get(D, row, col) *
							(gsl_matrix_get(H_core, row, col) + 
							 gsl_matrix_get(Fock, row, col));
	   }
	}

	return ene_elec;
}

//===============================
// form G matrix
//===============================
void form_G(int nbasis, gsl_matrix *D_prev, double *ERI, gsl_matrix *G)
{
	int mu, nu, lam, sig;
	for (mu=0; mu < nbasis; mu++)
	{
	   for (nu=0; nu < nbasis; nu++)
		{
			double val = 0.0;
	   		for (lam=0; lam < nbasis; lam++)
			{
	   			for (sig=0; sig < nbasis; sig++)
				{
					int mnls = ijkl2intindex(mu, nu, lam, sig);
					//int msln = ijkl2intindex(mu, sig, lam, nu);
					int mlns = ijkl2intindex(mu, lam, nu, sig);
					val += gsl_matrix_get(D_prev, lam, sig) * 
							(ERI[mnls] - 0.5 * ERI[mlns]);
				}
			}
			gsl_matrix_set(G, mu, nu, val);
		}
	}
}

//===============================
// form Fock matrix
//===============================
void form_Fock(int nbasis, gsl_matrix *H_core, gsl_matrix *G, gsl_matrix *Fock)
{
	int mu, nu;
	for (mu=0; mu < nbasis; mu++)
	{
		for (nu=0; nu < nbasis; nu++)
		{
			gsl_matrix_set(Fock, mu, nu, 
					gsl_matrix_get(H_core, mu, nu) + gsl_matrix_get(G, mu, nu));
		}
	}
}


// Generalized Wolfsberg-Helmholtz initial guess
void init_guess_GWH(Basis *p_basis, gsl_matrix *H_core, gsl_matrix *S, gsl_matrix *Fock)
{
	double cx = 1.0;
	int mu, nu;
	for (mu = 0; mu < p_basis->num; ++ mu)
	{
		double Hmm = gsl_matrix_get(H_core, mu, mu);
		for (nu = 0; nu < p_basis->num; ++ nu)
		{
			double Smn = gsl_matrix_get(S, mu, nu);
			double Hnn = gsl_matrix_get(H_core, nu, nu);
			double Fmn = cx * Smn * (Hmm + Hnn) / 2.0;
			gsl_matrix_set(Fock, mu, nu, Fmn);
		}
	}
}




// DIIS
void update_Fock_DIIS(int *p_diis_dim, int *p_diis_index, double *p_delta_DIIS, 
						gsl_matrix *Fock, gsl_matrix *D_prev, gsl_matrix *S, Basis *p_basis,
						double ***diis_err, double ***diis_Fock)
{
	int diis_dim = *p_diis_dim;
	int diis_index = *p_diis_index;
	double delta_DIIS;

	// dimension of DIIS, e.g. number of error matrices
	if (diis_dim < MAX_DIIS_DIM) { diis_dim = diis_index + 1; }

	// calculate FDS and SDF, using D_prev, Fock and S
	gsl_matrix *prod = gsl_matrix_alloc(p_basis->num, p_basis->num);
	gsl_matrix *FDS  = gsl_matrix_alloc(p_basis->num, p_basis->num);
	gsl_matrix *SDF  = gsl_matrix_alloc(p_basis->num, p_basis->num);

	gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, Fock, D_prev, 0.0, prod);
	gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, prod, S, 0.0, FDS);

	gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, S, D_prev, 0.0, prod);
	gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, prod, Fock, 0.0, SDF);

	gsl_matrix_free(prod);

	// new error matrix: e = FDS - SDF
	delta_DIIS = 0.0;
	int row, col;
	for (row=0; row < p_basis->num; row++)
	{
		for (col=0; col < p_basis->num; col++)
		{
			double err = gsl_matrix_get(FDS, row, col) - gsl_matrix_get(SDF, row, col);

			diis_err[diis_index][row][col] = err;
			diis_Fock[diis_index][row][col] = gsl_matrix_get(Fock, row, col);

			delta_DIIS += err * err;
		}
	}
	delta_DIIS = sqrt(delta_DIIS);

	gsl_matrix_free(FDS);
	gsl_matrix_free(SDF);

	// apply DIIS if there are two or more error matrices
	if (diis_dim > 1)
	{
		// construct B matrix and bb vector; B .dot. cc = bb
		gsl_matrix *B  = gsl_matrix_alloc(diis_dim + 1, diis_dim + 1);
		gsl_vector *bb = gsl_vector_alloc(diis_dim + 1);

		for (row = 0; row < diis_dim; ++ row)
		{
			for (col = 0; col < diis_dim; ++ col)
			{
				gsl_matrix_set (B, row, col,
						mat_inn_prod(p_basis->num, diis_err[row], diis_err[col]));
			}
		}

		int idiis;
		for (idiis = 0; idiis < diis_dim; ++ idiis)
		{
			gsl_matrix_set (B, diis_dim, idiis, -1.0);
			gsl_matrix_set (B, idiis, diis_dim, -1.0);
			gsl_vector_set (bb, idiis, 0.0);
		}

		gsl_matrix_set (B, diis_dim, diis_dim, 0.0);
		gsl_vector_set (bb, diis_dim, -1.0);

		// solve matrix equation; B .dot. cc = bb
		int ss;
		gsl_vector *cc = gsl_vector_alloc (diis_dim + 1);
		gsl_permutation *pp = gsl_permutation_alloc (diis_dim + 1);
		gsl_linalg_LU_decomp (B, pp, &ss);
		gsl_linalg_LU_solve (B, pp, bb, cc);
		gsl_permutation_free (pp);

		// update Fock matrix
		gsl_matrix_set_zero (Fock);
		for (idiis = 0; idiis < diis_dim; ++ idiis)
		{
			double ci = gsl_vector_get (cc, idiis);

			for (row = 0; row < p_basis->num; ++ row)
			{
				for (col = 0; col < p_basis->num; ++ col)
				{
					double Fab = gsl_matrix_get (Fock, row, col);
					Fab += ci * diis_Fock[idiis][row][col];
					gsl_matrix_set (Fock, row, col, Fab);
				}
			}
		}

		// free matrix B and vectors bb, cc
		gsl_matrix_free(B);
		gsl_vector_free(bb);
		gsl_vector_free(cc);
	}

	// update DIIS index, e.g. which error matrix to be updated
	++ diis_index;
	if (MAX_DIIS_DIM == diis_index) { diis_index = 0; }

	*p_diis_dim = diis_dim;
	*p_diis_index = diis_index;
	*p_delta_DIIS = delta_DIIS;
}

