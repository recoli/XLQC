/*****************************************************************************
 This file is part of the XLQC program.                                      
 Copyright (C) 2015 Xin Li <lixin.reco@gmail.com>                            
                                                                           
 Filename:  scf.cc                                                      
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
#include <string>

#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

#include "typedef.h"
#include "basis.h"
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
       fprintf(stderr, "Error: incorrect DIM in my_eigen_symmv!\n");
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
    for (int row = 0; row < A->size1; ++ row)
    {
        for (int col = 0; col < A->size2; ++ col)
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
    for (int row = 0; row < x->size; ++ row)
    {
        printf("%12.7f\n", gsl_vector_get(x, row));
    }
}

//===============================
// get core Hamiltonian
//===============================
void sum_H_core(int nbasis, gsl_matrix *H_core, gsl_matrix *T, gsl_matrix *V)
{
    for (int row = 0; row < nbasis; ++ row)
    {
       for (int col = 0; col < nbasis; ++ col)
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
    for (int row = 0; row < nbasis; ++ row)
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


    gsl_matrix_free (prod);
    gsl_matrix_free (Fock_p);
    gsl_matrix_free (Coef_p);
}

//===============================
// from MO coeffcients to density matrix
//===============================
void Coef_to_Dens(int nbasis, int n_occ, gsl_matrix *Coef, gsl_matrix *D)
{    
    for (int row=0; row < nbasis; ++ row)
    {
       for (int col=0; col < nbasis; ++ col)
        {
            double val = 0.0;
            for (int m = 0; m < n_occ; ++ m)
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
    for (int row = 0; row < nbasis; ++ row)
    {
       for (int col = 0; col < nbasis; ++ col)
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
void form_JK(int nbasis, gsl_matrix *D_prev, double *ERI, gsl_matrix *J, gsl_matrix *K)
{
    for (int mu = 0; mu < nbasis; ++ mu)
    {
        for (int nu = 0; nu <= mu; ++ nu)
        {
            double val_J = 0.0;
            double val_K = 0.0;
            for (int lam = 0; lam < nbasis; ++ lam)
            {
                for (int sig = 0; sig < nbasis; ++ sig)
                {
                    int mnls = ijkl2intindex(mu, nu, lam, sig);
                    int mlns = ijkl2intindex(mu, lam, nu, sig);

                    double D_ls = gsl_matrix_get(D_prev, lam, sig);
                    val_J += D_ls * ERI[mnls];
                    val_K += D_ls * ERI[mlns];
                }
            }
            gsl_matrix_set(J, mu, nu, val_J);
            gsl_matrix_set(K, mu, nu, val_K);

            if (nu != mu)
            {
                gsl_matrix_set(J, nu, mu, val_J);
                gsl_matrix_set(K, nu, mu, val_K);
            }
        }
    }
}

//===============================
// form Fock matrix
//===============================
void form_Fock(int nbasis, gsl_matrix *H_core, gsl_matrix *J, gsl_matrix *K, gsl_matrix *Fock)
{
    for (int mu = 0; mu < nbasis; ++ mu)
    {
        for (int nu = 0; nu < nbasis; ++ nu)
        {
            gsl_matrix_set(Fock, mu, nu, 
                gsl_matrix_get(H_core, mu, nu) + gsl_matrix_get(J, mu, nu) 
                - 0.5 * gsl_matrix_get(K, mu, nu));
        }
    }
}


// Generalized Wolfsberg-Helmholtz initial guess
void init_guess_GWH(Basis *p_basis, gsl_matrix *H_core, gsl_matrix *S, gsl_matrix *Fock)
{
    for (int mu = 0; mu < p_basis->num; ++ mu)
    {
        double Hmm = gsl_matrix_get(H_core, mu, mu);
        for (int nu = 0; nu < p_basis->num; ++ nu)
        {
            double Smn = gsl_matrix_get(S, mu, nu);
            double Hnn = gsl_matrix_get(H_core, nu, nu);
            double Fmn = Smn * (Hmm + Hnn) / 2.0;
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
    for (int row = 0; row < p_basis->num; ++ row)
    {
        for (int col=0; col < p_basis->num; ++ col)
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

        for (int row = 0; row < diis_dim; ++ row)
        {
            for (int col = 0; col < diis_dim; ++ col)
            {
                gsl_matrix_set (B, row, col,
                    mat_inn_prod(p_basis->num, diis_err[row], diis_err[col]));
            }
        }

        for (int idiis = 0; idiis < diis_dim; ++ idiis)
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
        for (int idiis = 0; idiis < diis_dim; ++ idiis)
        {
            double ci = gsl_vector_get (cc, idiis);

            for (int row = 0; row < p_basis->num; ++ row)
            {
                for (int col = 0; col < p_basis->num; ++ col)
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


// direct SCF
void direct_form_G(Basis *p_basis, gsl_matrix *D_prev, gsl_matrix *Q, gsl_matrix *G)
{
    gsl_matrix_set_zero(G);

    for (int a = 0; a < p_basis->num; ++ a)
    {
        for (int b = 0; b <= a; ++ b)
        {
            //int ij = ij2intindex(a, b);

            double Qab = gsl_matrix_get(Q,a,b);

            double Dmax = 0.0;
            double Dab = gsl_matrix_get(D_prev,a,b);
            if (fabs(2.0 * Dab) > Dmax) { Dmax = fabs(2.0 * Dab); }

            for (int c = 0; c <= a; ++ c)
            {
                double half_Dac = 0.5 * gsl_matrix_get(D_prev,a,c);
                double half_Dbc = 0.5 * gsl_matrix_get(D_prev,b,c);
                if (fabs(half_Dac) > Dmax) { Dmax = fabs(half_Dac); }
                if (fabs(half_Dbc) > Dmax) { Dmax = fabs(half_Dbc); }

                int d_max = (a == c) ? b : c;
                for (int d = 0; d <= d_max; ++ d)
                {
                    //int kl = ij2intindex(c, d);
                    //if (ij < kl) { continue; }

                    double half_Dad = 0.5 * gsl_matrix_get(D_prev,a,d);
                    double half_Dbd = 0.5 * gsl_matrix_get(D_prev,b,d);
                    double Dcd = gsl_matrix_get(D_prev,c,d);
                    if (fabs(half_Dad) > Dmax) { Dmax = fabs(half_Dad); }
                    if (fabs(half_Dbd) > Dmax) { Dmax = fabs(half_Dbd); }
                    if (fabs(2.0 * Dcd) > Dmax) { Dmax = fabs(2.0 * Dcd); }

                    // Schwarz inequality
                    // (ab|cd) <= sqrt(ab|ab) * sqrt(cd|cd)
                    double Qcd = gsl_matrix_get(Q,c,d);
                    if (Qab * Qcd * Dmax < 1.0e-10) { continue; }


                    double eri = calc_int_eri_rys(p_basis, a, b, c, d);

                    double Dab_eri = Dab * eri;
                    double Dcd_eri = Dcd * eri;
                    double half_Dac_eri = half_Dac * eri;
                    double half_Dad_eri = half_Dad * eri;
                    double half_Dbc_eri = half_Dbc * eri;
                    double half_Dbd_eri = half_Dbd * eri;

                    // ab|cd  -->  G_ab += D_cd * ERI_abcd
                    // ab|cd  -->  G_ac -= 0.5 * D_bd * ERI_abcd
                    gsl_matrix_set(G,a,b, gsl_matrix_get(G,a,b) + Dcd_eri);
                    gsl_matrix_set(G,a,c, gsl_matrix_get(G,a,c) - half_Dbd_eri);

                    // ab|dc  -->  G_ab += D_dc * ERI_abcd
                    // ab|dc  -->  G_ad -= 0.5 * D_bc * ERI_abcd
                    if (c != d)
                    {
                        gsl_matrix_set(G,a,b, gsl_matrix_get(G,a,b) + Dcd_eri);
                        gsl_matrix_set(G,a,d, gsl_matrix_get(G,a,d) - half_Dbc_eri);
                    }

                    // ba|cd  -->  G_ba += D_cd * ERI_abcd
                    // ba|cd  -->  G_bc -= 0.5 * D_ad * ERI_abcd
                    if (a != b)
                    {
                        gsl_matrix_set(G,b,a, gsl_matrix_get(G,b,a) + Dcd_eri);
                        gsl_matrix_set(G,b,c, gsl_matrix_get(G,b,c) - half_Dad_eri);

                        // ba|dc  -->  G_ba += D_dc * ERI_abcd
                        // ba|dc  -->  G_bd -= 0.5 * D_ac * ERI_abcd
                        if (c != d)
                        {
                            gsl_matrix_set(G,b,a, gsl_matrix_get(G,b,a) + Dcd_eri);
                            gsl_matrix_set(G,b,d, gsl_matrix_get(G,b,d) - half_Dac_eri);
                        }
                    }

                    // ab<==>cd permutations
                    //if (ij != kl)
                    if (a != c || b != d)
                    {
                        gsl_matrix_set(G,c,d, gsl_matrix_get(G,c,d) + Dab_eri);
                        gsl_matrix_set(G,c,a, gsl_matrix_get(G,c,a) - half_Dbd_eri);

                        if (a != b)
                        {
                            gsl_matrix_set(G,c,d, gsl_matrix_get(G,c,d) + Dab_eri);
                            gsl_matrix_set(G,c,b, gsl_matrix_get(G,c,b) - half_Dad_eri);
                        }

                        if (c != d)
                        {
                            gsl_matrix_set(G,d,c, gsl_matrix_get(G,d,c) + Dab_eri);
                            gsl_matrix_set(G,d,a, gsl_matrix_get(G,d,a) - half_Dbc_eri);

                            if (a != b)
                            {
                                gsl_matrix_set(G,d,c, gsl_matrix_get(G,d,c) + Dab_eri);
                                gsl_matrix_set(G,d,b, gsl_matrix_get(G,d,b) - half_Dac_eri);
                            }
                        }
                    }
                }
            }
        }
    }
}


// sqrt(ab|ab) for prescreening 
void form_Q(Basis *p_basis, gsl_matrix *Q)
{
    for (int a = 0; a < p_basis->num; ++ a)
    {
        for (int b = 0; b <= a; ++ b)
        {
            double eri = calc_int_eri_rys(p_basis, a, b, a, b);
            double Qab = sqrt(eri);
            gsl_matrix_set(Q, a, b, Qab);
            if (a != b) { gsl_matrix_set(Q, b, a, Qab); }
        }
    }
}
