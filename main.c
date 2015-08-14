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
#include "basis.h"
#include "scf.h"

#define CART_DIM 3
#define MAX_DIIS_DIM 6

int main(int argc, char* argv[])
{
	//====== parse geom.dat ========

	// get number of atoms
	int natoms = get_natoms();
	printf("Natoms = %d\n", natoms);

	// atomic coordinates and atom name
	double **atom_pos = (double **)my_malloc(sizeof(double *) * natoms);
	char **atom_name = (char **)my_malloc(sizeof(char *) * natoms);

	int iatom;
	for (iatom = 0; iatom < natoms; ++ iatom)
	{
		atom_pos[iatom] = (double *)my_malloc(sizeof(double) * CART_DIM);
		atom_name[iatom] = (char *)my_malloc(sizeof(char) * 5);
	}

	// nuclear charge
	int *atom_nuc_chg = (int *)my_malloc(sizeof(int) * natoms);

	// read atomic positions, nuclear charge and atom name
	read_geom(atom_pos, atom_nuc_chg, atom_name);

	for (iatom = 0; iatom < natoms; ++ iatom)
	{
		printf("%s (%.1f)  %.10f  %.10f  %.10f\n", atom_name[iatom], (double)atom_nuc_chg[iatom],
				atom_pos[iatom][0], atom_pos[iatom][1], atom_pos[iatom][2]);
	}
	

	//====== parse basis.dat ========

	// parse basis functions
	int nbasis;
	parse_basis(natoms, atom_name, atom_nuc_chg, &nbasis);

	printf("System Nbasis = %d\n", nbasis);

	// basis function exponents, coefficients, and center positions
	double **expon, **coef, **xbas;
	expon = (double **)my_malloc(sizeof(double *) * nbasis);
	coef  = (double **)my_malloc(sizeof(double *) * nbasis);
	xbas  = (double **)my_malloc(sizeof(double *) * nbasis);

	// number of primitive functions in each contracted funciton
	int *nprims;
	nprims = (int *)my_malloc(sizeof(int) * nbasis);

	// Cartesian l,m,n
	int **lmn;
	lmn = (int **)my_malloc(sizeof(int *) * nbasis);

	int ibasis;
	for (ibasis = 0; ibasis < nbasis; ++ ibasis)
	{
		xbas[ibasis] = (double *)my_malloc(sizeof(double) * CART_DIM);
		lmn[ibasis]  = (int *)my_malloc(sizeof(int) * CART_DIM);
	}

	// read basis set
	read_basis(natoms, atom_pos, nbasis, expon, coef, nprims, lmn, xbas);

	// normalization factors for each primitive function
	double **norm = (double **)my_malloc(sizeof(double *) * nbasis);

	for (ibasis = 0; ibasis < nbasis; ++ ibasis)
	{
		norm[ibasis] = (double *)my_malloc(sizeof(double) * nprims[ibasis]);

		int iprim;
		for (iprim = 0; iprim < nprims[ibasis]; ++ iprim)
		{
			norm[ibasis][iprim] = norm_factor(expon[ibasis][iprim], 
									lmn[ibasis][0], lmn[ibasis][1], lmn[ibasis][2]);
		}
	}


#ifdef DEBUG
	for (ibasis = 0; ibasis < nbasis; ++ ibasis)
	{
		int iprim;
		for (iprim = 0; iprim < nprims[ibasis]; ++ iprim)
		{
			printf("%16.8f%16.8f\n", expon[ibasis][iprim], coef[ibasis][iprim]);
		}
	}
#endif


	//====== nuclear repulsion energy ========

	double ene_nucl = 0.0;
	int ata, atb;
	for (ata = 0; ata < natoms; ++ ata)
	{
		for (atb = ata + 1; atb < natoms; ++ atb)
		{
			double dx = atom_pos[ata][0] - atom_pos[atb][0];
			double dy = atom_pos[ata][1] - atom_pos[atb][1];
			double dz = atom_pos[ata][2] - atom_pos[atb][2];
			double dr = sqrt(dx*dx + dy*dy + dz*dz);
			ene_nucl += atom_nuc_chg[ata] * atom_nuc_chg[atb] / dr;
		}
	}

	fprintf(stdout, "Nuclear repulsion = %-20.10f\n", ene_nucl);


	//====== one- and two-electron integrals ========

	// overlap, kinetic energy and nuclear attraction integral
	gsl_matrix *S = gsl_matrix_alloc(nbasis, nbasis);
	gsl_matrix *T = gsl_matrix_alloc(nbasis, nbasis);
	gsl_matrix *V = gsl_matrix_alloc(nbasis, nbasis);

	// two-electron ingetral
	int n_combi = nbasis * (nbasis + 1) / 2;
	int n_eri = n_combi * (n_combi + 1) / 2;
	double *ERI = (double *)my_malloc(sizeof(double) * n_eri);


	int a,b,c,d;
	for (a = 0; a < nbasis; ++ a)
	{
		for (b = 0; b <= a; ++ b)
		{
			// overlap
			double s;
			s = contr_overlap(
				nprims[a], expon[a], coef[a], norm[a], xbas[a][0], xbas[a][1], xbas[a][2], lmn[a][0], lmn[a][1], lmn[a][2],
				nprims[b], expon[b], coef[b], norm[b], xbas[b][0], xbas[b][1], xbas[b][2], lmn[b][0], lmn[b][1], lmn[b][2]);

			// kinetic energy
			double t;
			t = contr_kinetic(
				nprims[a], expon[a], coef[a], norm[a], xbas[a][0], xbas[a][1], xbas[a][2], lmn[a][0], lmn[a][1], lmn[a][2],
				nprims[b], expon[b], coef[b], norm[b], xbas[b][0], xbas[b][1], xbas[b][2], lmn[b][0], lmn[b][1], lmn[b][2]);

			// nuclear repulsion
			double v = 0.0;
			for (c = 0; c < natoms; ++ c)
			{
				v += contr_nuc_attr(
					 nprims[a], expon[a], coef[a], norm[a], xbas[a][0], xbas[a][1], xbas[a][2], lmn[a][0], lmn[a][1], lmn[a][2],
					 nprims[b], expon[b], coef[b], norm[b], xbas[b][0], xbas[b][1], xbas[b][2], lmn[b][0], lmn[b][1], lmn[b][2],
					 atom_nuc_chg[c], atom_pos[c][0], atom_pos[c][1], atom_pos[c][2]);
			}


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


			int ij = ij2intindex(a, b);
			// two-electron integral
			for (c = 0; c < nbasis; ++ c)
			{
				for (d = 0; d <= c; ++ d)
				{
					int kl = ij2intindex(c, d);
					if (ij < kl) { continue; }

					int ijkl = ij2intindex(ij, kl);

					double eri;
					/*
					eri = rys_contr_coulomb(
						  nprims[a], expon[a], coef[a], norm[a], xbas[a][0], xbas[a][1], xbas[a][2], lmn[a][0], lmn[a][1], lmn[a][2],
						  nprims[b], expon[b], coef[b], norm[b], xbas[b][0], xbas[b][1], xbas[b][2], lmn[b][0], lmn[b][1], lmn[b][2],
						  nprims[c], expon[c], coef[c], norm[c], xbas[c][0], xbas[c][1], xbas[c][2], lmn[c][0], lmn[c][1], lmn[c][2],
						  nprims[d], expon[d], coef[d], norm[d], xbas[d][0], xbas[d][1], xbas[d][2], lmn[d][0], lmn[d][1], lmn[d][2]);
					*/

					// use HGP for two-electron integrals
					eri = contr_hrr(
						  nprims[a], xbas[a][0], xbas[a][1], xbas[a][2], norm[a], lmn[a][0], lmn[a][1], lmn[a][2], expon[a], coef[a],
						  nprims[b], xbas[b][0], xbas[b][1], xbas[b][2], norm[b], lmn[b][0], lmn[b][1], lmn[b][2], expon[b], coef[b],
						  nprims[c], xbas[c][0], xbas[c][1], xbas[c][2], norm[c], lmn[c][0], lmn[c][1], lmn[c][2], expon[c], coef[c],
						  nprims[d], xbas[d][0], xbas[d][1], xbas[d][2], norm[d], lmn[d][0], lmn[d][1], lmn[d][2], expon[d], coef[d]);

					ERI[ijkl] = eri;
				}
			}
		}
	}


	//====== start SCF calculation ========

	// NOTE: assume zero charge and closed-shell electronics structure
	int n_elec = 0;
	for (iatom = 0; iatom < natoms; ++ iatom)
	{
		n_elec += atom_nuc_chg[iatom];
	}

	if (n_elec% 2 != 0)
	{
		fprintf(stderr, "Error: Number of electrons (%d) is not even!\n", n_elec);
	}

	int n_occ = n_elec / 2;


	// get core Hamiltonian
	gsl_matrix *H_core = gsl_matrix_alloc(nbasis, nbasis);
	sum_H_core(nbasis, H_core, T, V);

	// get S^-1/2
	gsl_matrix *S_invsqrt = gsl_matrix_alloc(nbasis, nbasis);
	diag_overlap(nbasis, S, S_invsqrt);


#ifdef DEBUG
	printf("S:\n"); my_print_matrix(S);
	printf("T:\n"); my_print_matrix(T);
	printf("V:\n"); my_print_matrix(V);
	printf("H_core:\n"); my_print_matrix(H_core);
	printf("S^-1/2:\n"); my_print_matrix(S_invsqrt);
#endif


	// matrices, vector and variables to be used in SCF
	gsl_matrix *D_prev = gsl_matrix_alloc(nbasis, nbasis);
	gsl_matrix *G = gsl_matrix_alloc(nbasis, nbasis);
	gsl_matrix *Fock = gsl_matrix_alloc(nbasis, nbasis);
	gsl_matrix *Coef = gsl_matrix_alloc(nbasis, nbasis);
	gsl_matrix *D = gsl_matrix_alloc(nbasis, nbasis);
	gsl_vector *emo = gsl_vector_alloc(nbasis);
	double ene_elec, ene_total, ene_prev;

	// initialize density matrix
	gsl_matrix_set_zero(D_prev);
	ene_prev = 0.0;

	fprintf(stdout, "%5s %20s %20s %20s %20s\n",
			"Iter", "E_total", "delta_E", "rms_D", "delta_DIIS");



	// DIIS error and Fock matrices
	double ***diis_err  = (double ***)my_malloc(sizeof(double **) * MAX_DIIS_DIM);
	double ***diis_Fock = (double ***)my_malloc(sizeof(double **) * MAX_DIIS_DIM);
	int idiis;
	for (idiis = 0; idiis < MAX_DIIS_DIM; ++ idiis)
	{
		diis_err[idiis]  = (double **)my_malloc(sizeof(double *) * nbasis);
		diis_Fock[idiis] = (double **)my_malloc(sizeof(double *) * nbasis);
		for (ibasis = 0; ibasis < nbasis; ++ ibasis)
		{
			diis_err[idiis][ibasis]  = (double *)my_malloc(sizeof(double) * nbasis);
			diis_Fock[idiis][ibasis] = (double *)my_malloc(sizeof(double) * nbasis);
		}
	}

	// DIIS index and dimension
	int diis_index = 0;
	int diis_dim = 0;

	double delta_DIIS;

	// gsl matrices used in DIIS
	gsl_matrix *prod = gsl_matrix_alloc(nbasis, nbasis);
	gsl_matrix *FDS  = gsl_matrix_alloc(nbasis, nbasis);
	gsl_matrix *SDF  = gsl_matrix_alloc(nbasis, nbasis);


	// Generalized Wolfsberg-Helmholtz initial guess
	double cx = 1.0;
	int mu, nu;
	for (mu = 0; mu < nbasis; ++ mu)
	{
		double Hmm = gsl_matrix_get(H_core, mu, mu);
		for (nu = 0; nu < nbasis; ++ nu)
		{
			double Smn = gsl_matrix_get(S, mu, nu);
			double Hnn = gsl_matrix_get(H_core, nu, nu);
			double Fmn = cx * Smn * (Hmm + Hnn) / 2.0;
			gsl_matrix_set(Fock, mu, nu, Fmn);
		}
	}

	Fock_to_Coef(nbasis, Fock, S_invsqrt, Coef, emo);
	Coef_to_Dens(nbasis, n_occ, Coef, D_prev);


	int iter = 0;
	while(1)
	{
		// SCF procedure:
		// Form new Fock matrix
		// F' = S^-1/2 * F * S^-1/2
		// diagonalize F' matrix to get C'
		// C = S^-1/2 * C'
		// compute new density matrix
		form_G(nbasis, D_prev, ERI, G);
		form_Fock(nbasis, H_core, G, Fock);


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
			for (row=0; row < nbasis; row++)
			{
				for (col=0; col < nbasis; col++)
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
								mat_inn_prod(nbasis, diis_err[row], diis_err[col]));
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

					for (row = 0; row < nbasis; ++ row)
					{
						for (col = 0; col < nbasis; ++ col)
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


		Fock_to_Coef(nbasis, Fock, S_invsqrt, Coef, emo);
		Coef_to_Dens(nbasis, n_occ, Coef, D);

		ene_elec = get_elec_ene(nbasis, D, H_core, Fock);
		ene_total = ene_nucl + ene_elec;


#ifdef DEBUG
		printf("F:\n"); my_print_matrix(Fock);
		printf("C:\n"); my_print_matrix(Coef);
		printf("P:\n"); my_print_matrix(D);
#endif


		// check convergence
		double delta_E = ene_total - ene_prev;

		double rms_D = 0.0;
		int mu, nu;
		for (mu = 0; mu < nbasis; ++ mu)
		{
			for (nu = 0; nu < nbasis; ++ nu)
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
		if (fabs(delta_E) < 1.0e-12 &&
			rms_D < 1.0e-10 && delta_DIIS < 1.0e-10) { break; }


		// update energy and density matrix for the next iteration
		ene_prev = ene_total;
		gsl_matrix_memcpy(D_prev, D);

		// count iterations
		++ iter;
	}

	fprintf(stdout, "SCF converged! E_total = %20.10f\n", ene_total);


	fprintf(stdout, "%5s %10s %15s %12s\n", "MO", "State", "E(Eh)", "E(eV)");
	for (ibasis = 0; ibasis < nbasis; ++ ibasis)
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
		for (ibasis = 0; ibasis < nbasis; ++ ibasis)
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


	// free arrays for geometry
	for (iatom = 0; iatom < natoms; ++ iatom)
	{
		free(atom_pos[iatom]);
		free(atom_name[iatom]);
	}
	free(atom_pos);
	free(atom_name);

	free(atom_nuc_chg);


	// free arrays for basis set
	for (ibasis = 0; ibasis < nbasis; ++ ibasis)
	{
		free(expon[ibasis]);
		free(coef[ibasis]);
		free(xbas[ibasis]);
		free(lmn[ibasis]);
		free(norm[ibasis]);
	}
	free(expon);
	free(coef);
	free(xbas);
	free(lmn);
	free(norm);

	free(nprims);


	return 0;
}

