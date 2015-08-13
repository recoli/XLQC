#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>

#include "int_lib/cints.h"
#include "int_lib/crys.h"
#include "basis.h"
#include "scf.h"

#define CART_DIM 3

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


	//====== write enuc.dat ========

	// nuclear repulsion energy
	double enuc = 0.0;
	int ata, atb;
	for (ata = 0; ata < natoms; ++ ata)
	{
		for (atb = ata + 1; atb < natoms; ++ atb)
		{
			double dx = atom_pos[ata][0] - atom_pos[atb][0];
			double dy = atom_pos[ata][1] - atom_pos[atb][1];
			double dz = atom_pos[ata][2] - atom_pos[atb][2];
			double dr = sqrt(dx*dx + dy*dy + dz*dz);
			enuc += atom_nuc_chg[ata] * atom_nuc_chg[atb] / dr;
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
					eri = rys_contr_coulomb(
						  nprims[a], expon[a], coef[a], norm[a], xbas[a][0], xbas[a][1], xbas[a][2], lmn[a][0], lmn[a][1], lmn[a][2],
						  nprims[b], expon[b], coef[b], norm[b], xbas[b][0], xbas[b][1], xbas[b][2], lmn[b][0], lmn[b][1], lmn[b][2],
						  nprims[c], expon[c], coef[c], norm[c], xbas[c][0], xbas[c][1], xbas[c][2], lmn[c][0], lmn[c][1], lmn[c][2],
						  nprims[d], expon[d], coef[d], norm[d], xbas[d][0], xbas[d][1], xbas[d][2], lmn[d][0], lmn[d][1], lmn[d][2]);

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

	//double ene_nucl = get_nucl_ene();
	//int nbasis = get_nbasis();
	double ene_nucl = enuc;

	fprintf(stdout, "ENUC= %-22.12f\n", ene_nucl);
	fprintf(stdout, "NBASIS= %-18d\n", nbasis);


	// get core Hamiltonian
	gsl_matrix *H_core = gsl_matrix_alloc(nbasis, nbasis);
	sum_H_core(nbasis, H_core, T, V);

	// get S^-1/2
	gsl_matrix *S_invsqrt = gsl_matrix_alloc(nbasis, nbasis);
	diag_overlap(nbasis, S, S_invsqrt);


#ifdef DEBUG
	printf("\nS:\n"); my_print_matrix(S);
	printf("\nT:\n"); my_print_matrix(T);
	printf("\nV:\n"); my_print_matrix(V);
	printf("\nH_core:\n"); my_print_matrix(H_core);
	printf("\nS^-1/2:\n"); my_print_matrix(S_invsqrt);
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

	fprintf(stdout, "%5s %22s %22s %22s\n", "Iter", "E_total", "E_elec", "delta_E");

	int iter = 0;
	while(1)
	{
		iter++;

		// SCF procedure:
		// Form new Fock matrix
		// F' = S^-1/2 * F * S^-1/2
		// diagonalize F' matrix to get C'
		// C = S^-1/2 * C'
		// compute new density matrix
		form_G(nbasis, D_prev, ERI, G);
		form_Fock(nbasis, H_core, G, Fock);

		Fock_to_Coef(nbasis, Fock, S_invsqrt, Coef, emo);
		Coef_to_Dens(nbasis, n_occ, Coef, D);

		ene_elec = get_elec_ene(nbasis, D, H_core, Fock);
		ene_total = ene_nucl + ene_elec;


#ifdef DEBUG
		printf("\nG:\n"); my_print_matrix(G);
		printf("\nF:\n"); my_print_matrix(Fock);
		printf("\nC:\n"); my_print_matrix(Coef);
		printf("\nP:\n"); my_print_matrix(D);
#endif


		double deltaE = ene_total - ene_prev;
		fprintf(stdout, "%5d %22.12f %22.12f %22.12f\n", 
				iter, ene_total, ene_elec, deltaE);

		// check convergence
		double deltaD = 0.0;
		int mu, nu;
		for (mu = 0; mu < nbasis; ++ mu)
		{
			for (nu = 0; nu < nbasis; ++ nu)
			{
				double dd = gsl_matrix_get(D, mu, nu) - gsl_matrix_get(D_prev, mu, nu);
				deltaD += dd * dd;
			}
		}
		deltaD = sqrt(deltaD / 4.0);

		if (deltaD < 1.0e-8) { break; }

		// update energy and density matrix for the next iteration
		ene_prev = ene_total;
		for (mu = 0; mu < nbasis; ++ mu)
		{
			for (nu = 0; nu < nbasis; ++ nu)
			{
				gsl_matrix_set(D_prev, mu, nu, gsl_matrix_get(D, mu, nu));
			}
		}
	}

	fprintf(stdout, "SCF converged! E_total = %22.12f\n", ene_total);


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

