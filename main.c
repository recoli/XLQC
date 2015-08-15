#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

#include "int_lib/cints.h"
#include "int_lib/crys.h"
#include "int_lib/chgp.h"

#include "typedef.h"
#include "basis.h"
#include "scf.h"

int main(int argc, char* argv[])
{
	Atom   *p_atom   = (Atom *)my_malloc(sizeof(Atom) * 1);
	Basis  *p_basis  = (Basis *)my_malloc(sizeof(Basis) * 1);

	//====== parse geom.dat ========

	// get number of atoms
	p_atom->num = get_natoms();
	printf("Natoms = %d\n", p_atom->num);

	// atomic coordinates and atom name
	p_atom->pos = (double **)my_malloc(sizeof(double *) * p_atom->num);
	p_atom->name = (char **)my_malloc(sizeof(char *) * p_atom->num);

	int iatom;
	for (iatom = 0; iatom < p_atom->num; ++ iatom)
	{
		p_atom->pos[iatom] = (double *)my_malloc(sizeof(double) * CART_DIM);
		p_atom->name[iatom] = (char *)my_malloc(sizeof(char) * 5);
	}

	// nuclear charge
	p_atom->nuc_chg = (int *)my_malloc(sizeof(int) * p_atom->num);

	// read atomic positions, nuclear charge and atom name
	read_geom(p_atom);

	for (iatom = 0; iatom < p_atom->num; ++ iatom)
	{
		printf("%s (%.1f)  %.10f  %.10f  %.10f\n", 
				p_atom->name[iatom], (double)p_atom->nuc_chg[iatom],
				p_atom->pos[iatom][0], p_atom->pos[iatom][1], p_atom->pos[iatom][2]);
	}
	

	//====== parse basis.dat ========

	// parse basis functions
	parse_basis(p_atom, p_basis);

	printf("System Nbasis = %d\n", p_basis->num);

	// basis function exponents, coefficients, and normalization factors
	p_basis->expon = (double **)my_malloc(sizeof(double *) * p_basis->num);
	p_basis->coef  = (double **)my_malloc(sizeof(double *) * p_basis->num);
	p_basis->norm  = (double **)my_malloc(sizeof(double *) * p_basis->num);

	// number of primitive functions in each contracted funciton
	p_basis->nprims = (int *)my_malloc(sizeof(int) * p_basis->num);

	// Cartesian coordinates and l,m,n numbers
	p_basis->xbas  = (double **)my_malloc(sizeof(double *) * p_basis->num);
	p_basis->lmn = (int **)my_malloc(sizeof(int *) * p_basis->num);

	int ibasis;
	for (ibasis = 0; ibasis < p_basis->num; ++ ibasis)
	{
		p_basis->xbas[ibasis] = (double *)my_malloc(sizeof(double) * CART_DIM);
		p_basis->lmn[ibasis]  = (int *)my_malloc(sizeof(int) * CART_DIM);
	}

	// read basis set (also calculate normalization factors)
	read_basis(p_atom, p_basis);

#ifdef DEBUG
	for (ibasis = 0; ibasis < p_basis->num; ++ ibasis)
	{
		int iprim;
		for (iprim = 0; iprim < p_basis->nprims[ibasis]; ++ iprim)
		{
			printf("%16.8f%16.8f\n", 
					p_basis->expon[ibasis][iprim], p_basis->coef[ibasis][iprim]);
		}
	}
#endif


	//====== nuclear repulsion energy ========

	double ene_nucl = 0.0;
	int ata, atb;
	for (ata = 0; ata < p_atom->num; ++ ata)
	{
		for (atb = ata + 1; atb < p_atom->num; ++ atb)
		{
			double dx = p_atom->pos[ata][0] - p_atom->pos[atb][0];
			double dy = p_atom->pos[ata][1] - p_atom->pos[atb][1];
			double dz = p_atom->pos[ata][2] - p_atom->pos[atb][2];
			double dr = sqrt(dx*dx + dy*dy + dz*dz);
			ene_nucl += (double)p_atom->nuc_chg[ata] * 
						(double)p_atom->nuc_chg[atb] / dr;
		}
	}

	fprintf(stdout, "Nuclear repulsion = %-20.10f\n", ene_nucl);


	//====== one- and two-electron integrals ========

	// overlap, kinetic energy and nuclear attraction integral
	gsl_matrix *S = gsl_matrix_alloc(p_basis->num, p_basis->num);
	gsl_matrix *T = gsl_matrix_alloc(p_basis->num, p_basis->num);
	gsl_matrix *V = gsl_matrix_alloc(p_basis->num, p_basis->num);

	// two-electron ingetral
	int n_combi = p_basis->num * (p_basis->num + 1) / 2;
	int n_eri = n_combi * (n_combi + 1) / 2;
	double *ERI = (double *)my_malloc(sizeof(double) * n_eri);

	int a,b,c,d;
	for (a = 0; a < p_basis->num; ++ a)
	{
		for (b = 0; b <= a; ++ b)
		{
			// overlap
			double s = calc_int_overlap(p_basis, a, b);

			// kinetic energy
			double t = calc_int_kinetic(p_basis, a, b);

			// nuclear repulsion
			double v = calc_int_nuc_attr(p_basis, a, b, p_atom);

			// save one-electron integrals in matrices
			gsl_matrix_set(S, a, b, s);
			gsl_matrix_set(T, a, b, t);
			gsl_matrix_set(V, a, b, v);
			if (a != b)
			{
				gsl_matrix_set(S, b, a, s);
				gsl_matrix_set(T, b, a, t);
				gsl_matrix_set(V, b, a, v);
			}

			// two-electron integral
			int ij = ij2intindex(a, b);
			for (c = 0; c < p_basis->num; ++ c)
			{
				for (d = 0; d <= c; ++ d)
				{
					int kl = ij2intindex(c, d);
					if (ij < kl) { continue; }

					int ijkl = ij2intindex(ij, kl);

					// use HGP for two-electron integrals
					double eri = calc_int_eri_hgp(p_basis, a, b, c, d);

					ERI[ijkl] = eri;
				}
			}
		}
	}


	//====== start SCF calculation ========

	// NOTE: assume zero charge and closed-shell electronics structure
	int n_elec = 0;
	for (iatom = 0; iatom < p_atom->num; ++ iatom)
	{
		n_elec += p_atom->nuc_chg[iatom];
	}

	if (n_elec% 2 != 0)
	{
		fprintf(stderr, "Error: Number of electrons (%d) is not even!\n", n_elec);
	}

	int n_occ = n_elec / 2;


	// get core Hamiltonian
	gsl_matrix *H_core = gsl_matrix_alloc(p_basis->num, p_basis->num);
	sum_H_core(p_basis->num, H_core, T, V);

	// get S^-1/2
	gsl_matrix *S_invsqrt = gsl_matrix_alloc(p_basis->num, p_basis->num);
	diag_overlap(p_basis->num, S, S_invsqrt);


#ifdef DEBUG
	printf("S:\n"); my_print_matrix(S);
	printf("T:\n"); my_print_matrix(T);
	printf("V:\n"); my_print_matrix(V);
	printf("H_core:\n"); my_print_matrix(H_core);
	printf("S^-1/2:\n"); my_print_matrix(S_invsqrt);
#endif


	// matrices, vector and variables to be used in SCF
	gsl_matrix *D_prev = gsl_matrix_alloc(p_basis->num, p_basis->num);
	gsl_matrix *G = gsl_matrix_alloc(p_basis->num, p_basis->num);
	gsl_matrix *Fock = gsl_matrix_alloc(p_basis->num, p_basis->num);
	gsl_matrix *Coef = gsl_matrix_alloc(p_basis->num, p_basis->num);
	gsl_matrix *D = gsl_matrix_alloc(p_basis->num, p_basis->num);
	gsl_vector *emo = gsl_vector_alloc(p_basis->num);
	double ene_elec, ene_total, ene_prev;

	// initialize density matrix
	gsl_matrix_set_zero(D_prev);
	gsl_matrix_set_zero(D);
	ene_prev = 0.0;

	fprintf(stdout, "%5s %20s %20s %20s %20s\n",
			"Iter", "E_total", "delta_E", "rms_D", "delta_DIIS");



	// DIIS error and Fock matrices
	double ***diis_err  = (double ***)my_malloc(sizeof(double **) * MAX_DIIS_DIM);
	double ***diis_Fock = (double ***)my_malloc(sizeof(double **) * MAX_DIIS_DIM);
	int idiis;
	for (idiis = 0; idiis < MAX_DIIS_DIM; ++ idiis)
	{
		diis_err[idiis]  = (double **)my_malloc(sizeof(double *) * p_basis->num);
		diis_Fock[idiis] = (double **)my_malloc(sizeof(double *) * p_basis->num);
		for (ibasis = 0; ibasis < p_basis->num; ++ ibasis)
		{
			diis_err[idiis][ibasis]  = (double *)my_malloc(sizeof(double) * p_basis->num);
			diis_Fock[idiis][ibasis] = (double *)my_malloc(sizeof(double) * p_basis->num);
		}
	}

	// DIIS index and dimension
	int diis_index = 0;
	int diis_dim = 0;

	double delta_DIIS;

	// gsl matrices used in DIIS
	gsl_matrix *prod = gsl_matrix_alloc(p_basis->num, p_basis->num);
	gsl_matrix *FDS  = gsl_matrix_alloc(p_basis->num, p_basis->num);
	gsl_matrix *SDF  = gsl_matrix_alloc(p_basis->num, p_basis->num);


	// Generalized Wolfsberg-Helmholtz initial guess
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

	Fock_to_Coef(p_basis->num, Fock, S_invsqrt, Coef, emo);
	Coef_to_Dens(p_basis->num, n_occ, Coef, D_prev);


	/*
	// Q: sqrt(ab|ab) for prescreening of two-electron integrals
	gsl_matrix *Q = gsl_matrix_alloc(p_basis->num, p_basis->num);
	for (a = 0; a < p_basis->num; ++ a)
	{
		for (b = 0; b < p_basis->num; ++ b)
		{
			double eri = calc_int_eri_hgp(p_basis, a, b, a, b);
			double Qab = sqrt(eri);
			gsl_matrix_set(Q, a, b, Qab);
		}
	}
	*/


	// start SCF iterations
	int iter = 0;
	while(1)
	{
		// SCF procedure:
		// Form new Fock matrix
		// F' = S^-1/2 * F * S^-1/2
		// diagonalize F' matrix to get C'
		// C = S^-1/2 * C'
		// compute new density matrix


		form_G(p_basis->num, D_prev, ERI, G);
		/*
		gsl_matrix_set_zero(G);
		for (a = 0; a < p_basis->num; ++ a)
		{
			for (b = 0; b <= a; ++ b)
			{
				int ij = ij2intindex(a, b);

				double Qab = gsl_matrix_get(Q,a,b);

				for (c = 0; c < p_basis->num; ++ c)
				{
					for (d = 0; d <= c; ++ d)
					{
						int kl = ij2intindex(c, d);
						if (ij < kl) { continue; }


						// Schwarz inequality
						// (ab|cd) <= sqrt(ab|ab) * sqrt(cd|cd)
						double Qcd = gsl_matrix_get(Q,c,d);
						if (Qab * Qcd < 1.0e-8) { continue; }


						double eri = calc_int_eri_hgp(p_basis, a, b, c, d);

						// ab|cd  -->  G_ab += D_cd * ERI_abcd
						// ab|cd  -->  G_ac -= 0.5 * D_bd * ERI_abcd
						gsl_matrix_set(G,a,b, gsl_matrix_get(G,a,b) + gsl_matrix_get(D_prev,c,d) * eri);
						gsl_matrix_set(G,a,c, gsl_matrix_get(G,a,c) - 0.5 * gsl_matrix_get(D_prev,b,d) * eri);

						// ba|cd  -->  G_ba += D_cd * ERI_abcd
						// ba|cd  -->  G_bc -= 0.5 * D_ad * ERI_abcd
						if (a != b)
						{
							gsl_matrix_set(G,b,a, gsl_matrix_get(G,b,a) + gsl_matrix_get(D_prev,c,d) * eri);
							gsl_matrix_set(G,b,c, gsl_matrix_get(G,b,c) - 0.5 * gsl_matrix_get(D_prev,a,d) * eri);
						}

						// ab|dc  -->  G_ab += D_dc * ERI_abcd
						// ab|dc  -->  G_ad -= 0.5 * D_bc * ERI_abcd
						if (c != d)
						{
							gsl_matrix_set(G,a,b, gsl_matrix_get(G,a,b) + gsl_matrix_get(D_prev,d,c) * eri);
							gsl_matrix_set(G,a,d, gsl_matrix_get(G,a,d) - 0.5 * gsl_matrix_get(D_prev,b,c) * eri);
						}

						// ba|dc  -->  G_ba += D_dc * ERI_abcd
						// ba|dc  -->  G_bd -= 0.5 * D_ac * ERI_abcd
						if (a != b && c != d)
						{
							gsl_matrix_set(G,b,a, gsl_matrix_get(G,b,a) + gsl_matrix_get(D_prev,d,c) * eri);
							gsl_matrix_set(G,b,d, gsl_matrix_get(G,b,d) - 0.5 * gsl_matrix_get(D_prev,a,c) * eri);
						}

						// ab<==>cd permutations
						if (ij != kl)
						{
							gsl_matrix_set(G,c,d, gsl_matrix_get(G,c,d) + gsl_matrix_get(D_prev,a,b) * eri);
							gsl_matrix_set(G,c,a, gsl_matrix_get(G,c,a) - 0.5 * gsl_matrix_get(D_prev,d,b) * eri);

							if (c != d)
							{
								gsl_matrix_set(G,d,c, gsl_matrix_get(G,d,c) + gsl_matrix_get(D_prev,a,b) * eri);
								gsl_matrix_set(G,d,a, gsl_matrix_get(G,d,a) - 0.5 * gsl_matrix_get(D_prev,c,b) * eri);
							}

							if (a != b)
							{
								gsl_matrix_set(G,c,d, gsl_matrix_get(G,c,d) + gsl_matrix_get(D_prev,b,a) * eri);
								gsl_matrix_set(G,c,b, gsl_matrix_get(G,c,b) - 0.5 * gsl_matrix_get(D_prev,d,a) * eri);
							}

							if (c != d && a != b)
							{
								gsl_matrix_set(G,d,c, gsl_matrix_get(G,d,c) + gsl_matrix_get(D_prev,b,a) * eri);
								gsl_matrix_set(G,d,b, gsl_matrix_get(G,d,b) - 0.5 * gsl_matrix_get(D_prev,c,a) * eri);
							}
						}
					}
				}
			}
		}
		*/

#ifdef DEBUG
		printf("G:\n");
		my_print_matrix(G);
#endif

		form_Fock(p_basis->num, H_core, G, Fock);


		// start DIIS
		if (iter > 0)
		{
			// dimension of DIIS, e.g. number of error matrices
			if (diis_dim < MAX_DIIS_DIM) { diis_dim = diis_index + 1; }

			// calculate FDS and SDF, using D_prev, Fock and S
			gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, Fock, D_prev, 0.0, prod);
			gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, prod, S, 0.0, FDS);

			gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, S, D_prev, 0.0, prod);
			gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, prod, Fock, 0.0, SDF);

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
		}


		Fock_to_Coef(p_basis->num, Fock, S_invsqrt, Coef, emo);
		Coef_to_Dens(p_basis->num, n_occ, Coef, D);

		ene_elec = get_elec_ene(p_basis->num, D, H_core, Fock);
		ene_total = ene_nucl + ene_elec;


#ifdef DEBUG
		printf("F:\n"); my_print_matrix(Fock);
		printf("C:\n"); my_print_matrix(Coef);
		printf("P:\n"); my_print_matrix(D);
#endif


		// check convergence
		double delta_E = ene_total - ene_prev;

		double rms_D = 0.0;
		//int mu, nu;
		for (mu = 0; mu < p_basis->num; ++ mu)
		{
			for (nu = 0; nu < p_basis->num; ++ nu)
			{
				double dd = gsl_matrix_get(D, mu, nu) - gsl_matrix_get(D_prev, mu, nu);
				rms_D += dd * dd;
			}
		}
		rms_D = sqrt(rms_D);

		fprintf(stdout, "%5d %20.10f", iter, ene_total);
		if (iter > 0) { fprintf(stdout, " %20.10f %20.10f", delta_E, rms_D); }
		if (iter > 1) { fprintf(stdout, " %20.10f", delta_DIIS); }
		fprintf(stdout, "\n");


		// convergence criteria
		if (fabs(delta_E) < 1.0e-10 &&
			rms_D < 1.0e-8 && delta_DIIS < 1.0e-8) { break; }


		// update energy and density matrix for the next iteration
		ene_prev = ene_total;
		gsl_matrix_memcpy(D_prev, D);

		// count iterations
		++ iter;
	}

	fprintf(stdout, "SCF converged! E_total = %20.10f\n", ene_total);


	fprintf(stdout, "%5s %10s %15s %12s\n", "MO", "State", "E(Eh)", "E(eV)");
	for (ibasis = 0; ibasis < p_basis->num; ++ ibasis)
	{
		char occ[10];
		if (ibasis < n_occ) { strcpy(occ, "occ."); }
		else { strcpy(occ, "virt."); }

		// CODATA 2014: 1 Hartree = 27.21138602 eV
		double ener = gsl_vector_get(emo, ibasis);
		fprintf(stdout, "%5d %10s %15.5f %12.2f\n",
				ibasis + 1, occ, ener, ener * 27.21138602);
	}


	// free DIIS error and Fock matrices
	for (idiis = 0; idiis < MAX_DIIS_DIM; ++ idiis)
	{
		for (ibasis = 0; ibasis < p_basis->num; ++ ibasis)
		{
			free(diis_err[idiis][ibasis]);
			free(diis_Fock[idiis][ibasis]);
		}
		free(diis_err[idiis]);
		free(diis_Fock[idiis]);
	}
	free(diis_err);
	free(diis_Fock);

	// free intermediate matrices for DIIS
	gsl_matrix_free(prod);
	gsl_matrix_free(FDS);
	gsl_matrix_free(SDF);


	// free arrays for one- and two-electron integral
	gsl_matrix_free(S);
	gsl_matrix_free(T);
	gsl_matrix_free(V);
	free(ERI);

	//gsl_matrix_free(Q);


	// free arrays for geometry
	for (iatom = 0; iatom < p_atom->num; ++ iatom)
	{
		free(p_atom->pos[iatom]);
		free(p_atom->name[iatom]);
	}
	free(p_atom->pos);
	free(p_atom->name);

	free(p_atom->nuc_chg);


	// free arrays for basis set
	for (ibasis = 0; ibasis < p_basis->num; ++ ibasis)
	{
		free(p_basis->expon[ibasis]);
		free(p_basis->coef[ibasis]);
		free(p_basis->xbas[ibasis]);
		free(p_basis->lmn[ibasis]);
		free(p_basis->norm[ibasis]);
	}
	free(p_basis->expon);
	free(p_basis->coef);
	free(p_basis->xbas);
	free(p_basis->lmn);
	free(p_basis->norm);

	free(p_basis->nprims);


	return 0;
}

