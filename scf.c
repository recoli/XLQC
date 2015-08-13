#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>

#include "int_lib/cints.h"

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
	for (row=0; row<A[0].size1; row++)
	{
		for (col=0; col<A[0].size2; col++)
		{
			printf("%12.7f", gsl_matrix_get(A, row, col));
		}
		printf("\n");
	}
}

//===============================
// read nuclear repulsion energy
//===============================
double get_nucl_ene(void)
{
	double enuc;
	FILE *input;
	input = fopen("enuc.dat", "r");
	fscanf(input, "%lf", &enuc);
	fclose(input);

	return enuc;
}
 
//===============================
// get number of basis functions
//===============================
int get_nbasis(void)
{
	int nbasis = 0;
	FILE *input;
	input = fopen("s.dat", "r");
	while(1)
	{
		int a, b;
		double val;
		if(fscanf(input, "%d%d%lf", &a, &b, &val) != EOF)
		{
			if(a > nbasis) nbasis = a;
			if(b > nbasis) nbasis = b;
		}
		else
		{
			break;
		}
	}
	fclose(input);

	return nbasis;
}

//===============================
// read overlap integral
//===============================
void read_overlap(int nbasis, gsl_matrix *S)
{
	FILE *input;
	input = fopen("s.dat", "r");

	int i, j;
	for(i=0; i < nbasis; i++)
	{
		for(j=0; j < nbasis; j++)
		{
			int a, b;
			double val;

			fscanf(input, "%d%d%lf", &a, &b, &val);
			gsl_matrix_set(S, a-1, b-1, val);

			if (a != b)
			{
				gsl_matrix_set(S, b-1, a-1, val);
			}
		}
	}

	fclose(input);
}

//===============================
// read kinetic energy integral
//===============================
void read_kinetic(int nbasis, gsl_matrix *T)
{
	FILE *input;
	input = fopen("t.dat", "r");
	int i, j;
	for(i=0; i < nbasis; i++)
	{
		for(j=0; j < nbasis; j++)
		{
			int a, b;
			double val;

			fscanf(input, "%d%d%lf", &a, &b, &val);
			gsl_matrix_set(T, a-1, b-1, val);

			if (a != b)
			{
				gsl_matrix_set(T, b-1, a-1, val);
			}
		}
	}
	fclose(input);
}

//===============================
// read nuclear attraction integral
//===============================
void read_potential(int nbasis, gsl_matrix *V)
{
	FILE *input;
	input = fopen("v.dat", "r");
	int i, j;
	for(i=0; i < nbasis; i++)
	{
		for(j=0; j < nbasis; j++)
		{
			int a, b;
			double val;

			fscanf(input, "%d%d%lf", &a, &b, &val);
			gsl_matrix_set(V, a-1, b-1, val);

			if (a != b)
			{
				gsl_matrix_set(V, b-1, a-1, val);
			}
		}
	}
	fclose(input);
}

//===============================
// read two-electron integrals
//===============================
void read_eri(int n_eri, double *ERI)
{
	int i;
	for (i=0; i < n_eri; i++)
	{
		ERI[i] = 0.0;
	}

	int a,b,c,d;
	double val;
	FILE *input;
	input = fopen("eri.dat", "r");
	while(fscanf(input, "%d%d%d%d%lf", &a, &b, &c, &d, &val) != EOF)
	{
		int ijkl = ijkl2intindex(a-1, b-1, c-1, d-1);
		if(ijkl >= n_eri)
		{
			printf("Error: ijkl >= n_eri in read_eri\n");
			exit(1);
		}
		ERI[ijkl] = val;
	}
	fclose(input);
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

#ifdef DEBUG
	printf("\nF':\n"); my_print_matrix(Fock_p);
#endif

	// diagonalization of Fock_p
	// emo: eigenvalues
	// Coef_p: eigenvectors
	gsl_matrix *Coef_p = gsl_matrix_alloc(nbasis, nbasis);
	my_eigen_symmv(Fock_p, nbasis, emo, Coef_p);

	// C = S^-1/2 * C'
	gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, S_invsqrt, Coef_p, 0.0, Coef);

#ifdef DEBUG
	printf("\nC':\n"); my_print_matrix(Coef_p);
	printf("\nC:\n"); my_print_matrix(Coef);
#endif

	gsl_matrix *t1 = gsl_matrix_alloc(nbasis, nbasis);
	gsl_matrix *t2 = gsl_matrix_alloc(nbasis, nbasis);
	gsl_blas_dgemm (CblasTrans, CblasNoTrans, 1.0, Coef_p, Fock_p, 0.0, t1);
	gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, t1, Coef_p, 0.0, t2);

#ifdef DEBUG
	printf("\nC'T F' C':\n"); my_print_matrix(t2);
#endif

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
					int msln = ijkl2intindex(mu, sig, lam, nu);
					val += gsl_matrix_get(D_prev, lam, sig) * 
							(ERI[mnls] - 0.5 * ERI[msln]);
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

