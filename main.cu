/*****************************************************************************
 This file is part of the XLQC program.                                      
 Copyright (C) 2015 Xin Li <lixin.reco@gmail.com>                            
                                                                           
 Filename:  main.cu                                                      
 License:   BSD 3-Clause License

 This software is provided by the copyright holders and contributors "as is"
 and any express or implied warranties, including, but not limited to, the
 implied warranties of merchantability and fitness for a particular purpose are
 disclaimed. In no event shall the copyright holder or contributors be liable
 for any direct, indirect, incidental, special, exemplary, or consequential
 damages (including, but not limited to, procurement of substitute goods or
 services; loss of use, data, or profits; or business interruption) however
 caused and on any theory of liability, whether in contract, strict liability,
 or tort (including negligence or otherwise) arising in any way out of the use
 of this software, even if advised of the possibility of such damage.
 *****************************************************************************/

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>

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

#include "cuda_rys.h"

void my_cuda_safe(cudaError_t err, std::string word)
{
    if(err != cudaSuccess) 
    {
		fprintf(stderr, "Error during %s: ", word.c_str());

		// check for error
		cudaThreadSynchronize();
		cudaError_t error = cudaGetLastError();
		if(error != cudaSuccess)
		{
			// print the CUDA error message and exit
			fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
			exit(-1);
		}
    } 
}

int main(int argc, char* argv[])
{
	Atom   *p_atom   = (Atom *)my_malloc(sizeof(Atom) * 1);
	Basis  *p_basis  = (Basis *)my_malloc(sizeof(Basis) * 1);

	//====== parse geom.dat ========

	// get number of atoms
	p_atom->num = get_natoms();
	fprintf(stdout, "Natoms = %d\n", p_atom->num);

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

	fprintf(stdout, "Coordinates in atomic unit:\n");
	for (iatom = 0; iatom < p_atom->num; ++ iatom)
	{
		fprintf(stdout, "%s (%.1f)  %.10f  %.10f  %.10f\n", 
				p_atom->name[iatom], (double)p_atom->nuc_chg[iatom],
				p_atom->pos[iatom][0], p_atom->pos[iatom][1], p_atom->pos[iatom][2]);
	}
	
	// nuclear repulsion energy
	double ene_nucl = calc_ene_nucl(p_atom);
	fprintf(stdout, "Nuclear repulsion = %-20.10f\n", ene_nucl);


	//====== parse basis.dat ========

	// parse basis functions
	parse_basis(p_atom, p_basis);

	fprintf(stdout, "System Nbasis = %d\n", p_basis->num);

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
	print_basis(p_basis);
#endif


	//====== one- and two-electron integrals ========

	// overlap, kinetic energy and nuclear attraction integral
	gsl_matrix *S = gsl_matrix_alloc(p_basis->num, p_basis->num);
	gsl_matrix *T = gsl_matrix_alloc(p_basis->num, p_basis->num);
	gsl_matrix *V = gsl_matrix_alloc(p_basis->num, p_basis->num);

	// two-electron ingetral
	int n_combi = p_basis->num * (p_basis->num + 1) / 2;
	int n_eri = n_combi * (n_combi + 1) / 2;
	//fprintf(stdout, "N_eri = %d\n", n_eri);
	//double *ERI = (double *)my_malloc_2(sizeof(double) * n_eri, "ERI");

	int a,b;
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

			/*
			// two-electron integral
			int ij = ij2intindex(a, b);
			for (int c = 0; c <= a; ++ c)
			{
				int d_max = (a == c) ? b : c;
				for (int d = 0; d <= d_max; ++ d)
				{
					int kl = ij2intindex(c, d);
					//if (ij < kl) { continue; }

					int ijkl = ij2intindex(ij, kl);

					double eri = calc_int_eri_rys(p_basis, a, b, c, d);

					ERI[ijkl] = eri;
				}
			}
			*/
		}
	}


	// count number of primitive integrals in a <bra| or |ket>
	int count_prim = 0;
	int i,j;
	for (a = 0; a < p_basis->num; ++ a)
	{
		int lena = p_basis->nprims[a];
		for (b = 0; b <= a; ++ b)
		{
			int lenb = p_basis->nprims[b];
        
			for (i=0; i<lena; i++)
				for (j=0; j<lenb; j++)
					++ count_prim;
		}
	}

	// allocate memory for arrays on host
	size_t n_CI_bytes = sizeof(double) * n_combi;
	size_t n_CI_bytes_int = sizeof(int) * n_combi;
	size_t n_PI_bytes = sizeof(double) * count_prim;
	size_t n_ERI_bytes = sizeof(double) * n_eri;

	double *h_xa = (double *)my_malloc(n_CI_bytes);
	double *h_ya = (double *)my_malloc(n_CI_bytes);
	double *h_za = (double *)my_malloc(n_CI_bytes);
	int    *h_la = (int    *)my_malloc(n_CI_bytes_int);
	int    *h_ma = (int    *)my_malloc(n_CI_bytes_int);
	int    *h_na = (int    *)my_malloc(n_CI_bytes_int);
	double *h_aexps = (double *)my_malloc(n_PI_bytes);
	double *h_acoef = (double *)my_malloc(n_PI_bytes);
	// note that 'anorm' is absorbed into 'acoef'

	double *h_xb = (double *)my_malloc(n_CI_bytes);
	double *h_yb = (double *)my_malloc(n_CI_bytes);
	double *h_zb = (double *)my_malloc(n_CI_bytes);
	int    *h_lb = (int    *)my_malloc(n_CI_bytes_int);
	int    *h_mb = (int    *)my_malloc(n_CI_bytes_int);
	int    *h_nb = (int    *)my_malloc(n_CI_bytes_int);
	double *h_bexps = (double *)my_malloc(n_PI_bytes);
	double *h_bcoef = (double *)my_malloc(n_PI_bytes);
	// note that 'bnorm' is absorbed into 'bcoef'

	int *h_start_contr = (int *)my_malloc(n_CI_bytes_int);
	int *h_end_contr   = (int *)my_malloc(n_CI_bytes_int);

	double *h_eri = (double *)my_malloc(n_ERI_bytes);

	// fill arrays on host
	// index counts primitive integrals
	// index_contr counts contracted integrals
	int index = 0;
	int index_contr = 0;

	for (a = 0; a < p_basis->num; ++ a)
	{
		int lena = p_basis->nprims[a];
		for (b = 0; b <= a; ++ b)
		{
			int lenb = p_basis->nprims[b];

			h_start_contr[index_contr] = index;

			h_xa[index_contr] = p_basis->xbas[a][0];
			h_ya[index_contr] = p_basis->xbas[a][1];
			h_za[index_contr] = p_basis->xbas[a][2];

			h_la[index_contr] = p_basis->lmn[a][0];
			h_ma[index_contr] = p_basis->lmn[a][1];
			h_na[index_contr] = p_basis->lmn[a][2];
        
			h_xb[index_contr] = p_basis->xbas[b][0];
			h_yb[index_contr] = p_basis->xbas[b][1];
			h_zb[index_contr] = p_basis->xbas[b][2];
                            
			h_lb[index_contr] = p_basis->lmn[b][0];
			h_mb[index_contr] = p_basis->lmn[b][1];
			h_nb[index_contr] = p_basis->lmn[b][2];

			int i,j;
			for (i=0; i<lena; i++)
			{
				for (j=0; j<lenb; j++)
				{
					h_aexps[index] = p_basis->expon[a][i];
					h_acoef[index] = p_basis->coef[a][i] * p_basis->norm[a][i];
					// note that 'anorm' is absorbed into 'acoef'

					h_bexps[index] = p_basis->expon[b][j];
					h_bcoef[index] = p_basis->coef[b][j] * p_basis->norm[b][j];
					// note that 'bnorm' is absorbed into 'bcoef'

					++ index;
				}
			}

			h_end_contr[index_contr] = index - 1;

			++ index_contr;
		}
	}
	printf("Num_Prim_Combi  = %d (%d)\n", index, count_prim);
	printf("Num_Contr_Combi = %d (%d)\n", index_contr, n_combi);

	// initialize arrays on device
	double *dev_xa, *dev_ya, *dev_za;
	double *dev_xb, *dev_yb, *dev_zb;
	int    *dev_la, *dev_ma, *dev_na;
	int    *dev_lb, *dev_mb, *dev_nb;
	double *dev_aexps, *dev_acoef;
	double *dev_bexps, *dev_bcoef;

	dev_xa = NULL; dev_ya = NULL; dev_za = NULL;
	dev_xb = NULL; dev_yb = NULL; dev_zb = NULL;
	dev_la = NULL; dev_ma = NULL; dev_na = NULL;
	dev_lb = NULL; dev_mb = NULL; dev_nb = NULL;
	dev_aexps = NULL; dev_acoef = NULL;
	dev_bexps = NULL; dev_bcoef = NULL;

	int *dev_start_contr = NULL;
	int *dev_end_contr   = NULL;

	double *dev_eri = NULL;

	// allocate memories for arrays on device
	fprintf(stdout, "Mem_on_Device = %zu MB\n",
			(n_CI_bytes*8 + n_PI_bytes*4 + n_CI_bytes_int*6 + n_ERI_bytes) / 1000000);

	cudaMalloc((void**)&dev_xa, n_CI_bytes);
	cudaMalloc((void**)&dev_ya, n_CI_bytes);
	cudaMalloc((void**)&dev_za, n_CI_bytes);
	cudaMalloc((void**)&dev_xb, n_CI_bytes);
	cudaMalloc((void**)&dev_yb, n_CI_bytes);
	cudaMalloc((void**)&dev_zb, n_CI_bytes);

	if(dev_xa == NULL || dev_ya == NULL || dev_za == NULL ||
	   dev_xb == NULL || dev_yb == NULL || dev_zb == NULL)
	{
		printf("Error: cannot cudaMalloc for x_basis!\n");
		exit(1);
	}

	cudaMalloc((void**)&dev_la, n_CI_bytes_int);
	cudaMalloc((void**)&dev_ma, n_CI_bytes_int);
	cudaMalloc((void**)&dev_na, n_CI_bytes_int);
	cudaMalloc((void**)&dev_lb, n_CI_bytes_int);
	cudaMalloc((void**)&dev_mb, n_CI_bytes_int);
	cudaMalloc((void**)&dev_nb, n_CI_bytes_int);

	if(dev_la == NULL || dev_ma == NULL || dev_na == NULL ||
	   dev_lb == NULL || dev_mb == NULL || dev_nb == NULL)
	{
		printf("Error: cannot cudaMalloc for l_basis!\n");
		exit(1);
	}

	cudaMalloc((void**)&dev_aexps, n_PI_bytes);
	cudaMalloc((void**)&dev_acoef, n_PI_bytes);
	cudaMalloc((void**)&dev_bexps, n_PI_bytes);
	cudaMalloc((void**)&dev_bcoef, n_PI_bytes);

	if(dev_aexps == NULL || dev_acoef == NULL ||
	   dev_bexps == NULL || dev_bcoef == NULL)
	{
		printf("Error: cannot cudaMalloc for exp_basis!\n");
		exit(1);
	}

	cudaMalloc((void**)&dev_start_contr, n_CI_bytes);
	cudaMalloc((void**)&dev_end_contr,   n_CI_bytes);

	cudaMalloc((void**)&dev_eri, n_ERI_bytes);

	if(dev_eri == NULL || dev_start_contr == NULL || dev_end_contr == NULL)
	{
		printf("Error: cannot cudaMalloc for dev_eri!\n");
		exit(1);
	}

	// copy data from host to device
	my_cuda_safe(cudaMemcpy(dev_xa, h_xa, n_CI_bytes, cudaMemcpyHostToDevice),"mem_xa");
	my_cuda_safe(cudaMemcpy(dev_ya, h_ya, n_CI_bytes, cudaMemcpyHostToDevice),"mem_ya");
	my_cuda_safe(cudaMemcpy(dev_za, h_za, n_CI_bytes, cudaMemcpyHostToDevice),"mem_za");
	my_cuda_safe(cudaMemcpy(dev_xb, h_xb, n_CI_bytes, cudaMemcpyHostToDevice),"mem_xb");
	my_cuda_safe(cudaMemcpy(dev_yb, h_yb, n_CI_bytes, cudaMemcpyHostToDevice),"mem_yb");
	my_cuda_safe(cudaMemcpy(dev_zb, h_zb, n_CI_bytes, cudaMemcpyHostToDevice),"mem_zb");

	my_cuda_safe(cudaMemcpy(dev_la, h_la, n_CI_bytes_int, cudaMemcpyHostToDevice),"mem_la");
	my_cuda_safe(cudaMemcpy(dev_ma, h_ma, n_CI_bytes_int, cudaMemcpyHostToDevice),"mem_ma");
	my_cuda_safe(cudaMemcpy(dev_na, h_na, n_CI_bytes_int, cudaMemcpyHostToDevice),"mem_na");
	my_cuda_safe(cudaMemcpy(dev_lb, h_lb, n_CI_bytes_int, cudaMemcpyHostToDevice),"mem_lb");
	my_cuda_safe(cudaMemcpy(dev_mb, h_mb, n_CI_bytes_int, cudaMemcpyHostToDevice),"mem_mb");
	my_cuda_safe(cudaMemcpy(dev_nb, h_nb, n_CI_bytes_int, cudaMemcpyHostToDevice),"mem_nb");

	my_cuda_safe(cudaMemcpy(dev_aexps, h_aexps, n_PI_bytes, cudaMemcpyHostToDevice),"mem_ae");
	my_cuda_safe(cudaMemcpy(dev_acoef, h_acoef, n_PI_bytes, cudaMemcpyHostToDevice),"mem_ac");
	my_cuda_safe(cudaMemcpy(dev_bexps, h_bexps, n_PI_bytes, cudaMemcpyHostToDevice),"mem_be");
	my_cuda_safe(cudaMemcpy(dev_bcoef, h_bcoef, n_PI_bytes, cudaMemcpyHostToDevice),"mem_bc");

	my_cuda_safe(cudaMemcpy(dev_start_contr, h_start_contr, n_CI_bytes_int, cudaMemcpyHostToDevice),"mem_start");
	my_cuda_safe(cudaMemcpy(dev_end_contr,   h_end_contr,   n_CI_bytes_int, cudaMemcpyHostToDevice),"mem_end");


	// create 8x8 thread blocks
	dim3 block_size;
	block_size.x = 8;
	block_size.y = 8;

	// configure a two dimensional grid as well
	dim3 grid_size;
	grid_size.x = n_combi / block_size.x + (n_combi % block_size.x ? 1 : 0);
	grid_size.y = n_combi / block_size.y + (n_combi % block_size.y ? 1 : 0);


	// launch the kernel to calculate two-electron integrals on GPU
	cuda_rys_eri_2d<<<grid_size, block_size>>>
		(dev_xa,dev_ya,dev_za, dev_la,dev_ma,dev_na, dev_aexps,dev_acoef,
		 dev_xb,dev_yb,dev_zb, dev_lb,dev_mb,dev_nb, dev_bexps,dev_bcoef,
		 n_combi, dev_start_contr, dev_end_contr, dev_eri);

	// copy the results back to host
	my_cuda_safe(cudaMemcpy(h_eri, dev_eri, n_ERI_bytes, cudaMemcpyDeviceToHost),"mem_eri"); 


	/* just for test...
	int check_passed = 1;
	for (i = 0; i < n_eri; ++ i)
	{
		double diff = fabs(h_eri[i]-ERI[i]);
		if (diff > 1e-12)
		{
			check_passed = 0;
			printf("%-8d %18.12f %18.12f   %18.12f\n", i, ERI[i], h_eri[i], ERI[i]-h_eri[i]);
		}
	}
	if (check_passed) { printf("Check passed!\n"); }
	*/


	//====== start SCF calculation ========

	// NOTE: assume zero charge and closed-shell electronics structure
	int n_elec = 0;
	for (iatom = 0; iatom < p_atom->num; ++ iatom)
	{
		n_elec += p_atom->nuc_chg[iatom];
	}

	if (n_elec % 2 != 0)
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
	gsl_matrix *G      = gsl_matrix_alloc(p_basis->num, p_basis->num);
	gsl_matrix *Fock   = gsl_matrix_alloc(p_basis->num, p_basis->num);
	gsl_matrix *Coef   = gsl_matrix_alloc(p_basis->num, p_basis->num);
	gsl_matrix *D      = gsl_matrix_alloc(p_basis->num, p_basis->num);
	gsl_vector *emo    = gsl_vector_alloc(p_basis->num);
	double ene_elec, ene_total, ene_prev;

	// initialize density matrix
	gsl_matrix_set_zero(D_prev);
	gsl_matrix_set_zero(D);
	ene_prev = 0.0;


	// Generalized Wolfsberg-Helmholtz initial guess
	init_guess_GWH(p_basis, H_core, S, Fock);
	Fock_to_Coef(p_basis->num, Fock, S_invsqrt, Coef, emo);
	Coef_to_Dens(p_basis->num, n_occ, Coef, D_prev);


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

	fprintf(stdout, "%5s %20s %20s %20s %20s\n",
			"Iter", "E_total", "delta_E", "rms_D", "delta_DIIS");


	/*
	// Q: sqrt(ab|ab) for prescreening of two-electron integrals
	gsl_matrix *Q = gsl_matrix_alloc(p_basis->num, p_basis->num);
	form_Q(p_basis, Q);
	*/


	// start SCF iterations
	int iter = 0;
	while (1)
	{
		// SCF procedure:
		// Form new Fock matrix
		// F' = S^-1/2 * F * S^-1/2
		// diagonalize F' matrix to get C'
		// C = S^-1/2 * C'
		// compute new density matrix

		//form_G(p_basis->num, D_prev, ERI, G);
		//direct_form_G(p_basis, D_prev, Q, G);

		// use GPU-calculated two-electron integrals
		form_G(p_basis->num, D_prev, h_eri, G);

#ifdef DEBUG
		printf("G:\n"); my_print_matrix(G);
#endif

		form_Fock(p_basis->num, H_core, G, Fock);

		// DIIS
		if (iter > 0)
		{
			update_Fock_DIIS(&diis_dim, &diis_index, &delta_DIIS, 
						Fock, D_prev, S, p_basis, diis_err, diis_Fock);
		}

		// update density matrix and energies
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
		int mu, nu;
		for (mu = 0; mu < p_basis->num; ++ mu)
		{
			for (nu = 0; nu < p_basis->num; ++ nu)
			{
				double dd = gsl_matrix_get(D, mu, nu) - 
							gsl_matrix_get(D_prev, mu, nu);
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

	// SCF converged
	fprintf(stdout, "SCF converged! E_total = %20.10f\n", ene_total);

	// print MO information
	fprintf(stdout, "%5s %10s %15s %12s\n", "MO", "State", "E(Eh)", "E(eV)");
	for (ibasis = 0; ibasis < p_basis->num; ++ ibasis)
	{
		char occ[10];
		if (ibasis < n_occ) { strcpy(occ, "occ."); }
		else { strcpy(occ, "virt."); }

		double ener = gsl_vector_get(emo, ibasis);
		fprintf(stdout, "%5d %10s %15.5f %12.2f\n",
				ibasis + 1, occ, ener, ener * HARTREE2EV);
	}


	//====== free allocated memories ========

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

	// free arrays for one- and two-electron integral
	gsl_matrix_free(S);
	gsl_matrix_free(T);
	gsl_matrix_free(V);
	//free(ERI);

	//gsl_matrix_free(Q);

	// free matrices and vector for SCF
	gsl_matrix_free(H_core);
	gsl_matrix_free(S_invsqrt);
	gsl_matrix_free(D_prev);
	gsl_matrix_free(G);
	gsl_matrix_free(Fock);
	gsl_matrix_free(Coef);
	gsl_matrix_free(D);
	gsl_vector_free(emo);

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


	//====== the end of program ========

	return 0;
}
