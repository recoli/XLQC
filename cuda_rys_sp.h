/*****************************************************************************
 This file is part of the XLQC program.                                      
 Copyright (C) 2015 Xin Li <lixin.reco@gmail.com>                            
                                                                           
 Filename:  cuda_rys_sp.h                                                      
 License:   BSD 3-Clause License

 * The implementation of Rys quadrature routines in C is taken from the
 * PyQuante quantum chemistry program, Copyright (c) 2004, Richard P. Muller.
 * PyQuante version 1.2 and later is covered by the modified BSD license. 
 * Please see int_lib/LICENSE.
 
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

#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAXROOTS 7

void my_cuda_safe(cudaError_t err, std::string word);

__device__ int cuda_fact(int n);
__device__ int cuda_binomial(int a, int b);
__device__ void cuda_Roots(int n, float X, float roots[], float weights[]);
__device__ void cuda_Root123(int n, float X, float roots[], float weights[]);
__device__ void cuda_Root4(float X, float roots[], float weights[]);
__device__ void cuda_Root5(float X, float roots[], float weights[]);
__device__ void cuda_Root6(int n,float X, float roots[], float weights[]);
__device__ float cuda_Int1d(int i, int j,int k, int l,
                            float xi, float xj, float xk, float xl,
                            float alpha_ij_A, float alpha_kl_B, float sqrt_AB,
                            float A, float B, float Px, float Qx,
                            float inv_t1, float B00, float B1, float B1p, 
                            float G[][MAXROOTS]);
__device__ float cuda_rys_coulomb_repulsion(float xa,float ya,float za,float norma,
                                            int la,int ma,int na,float alphaa,
                                            float xb,float yb,float zb,float normb,
                                            int lb,int mb,int nb,float alphab,
                                            float xc,float yc,float zc,float normc,
                                            int lc,int mc,int nc,float alphac,
                                            float xd,float yd,float zd,float normd,
                                            int ld,int md,int nd,float alphad);
__global__ void cuda_rys_eri_2d(double *xa,double *ya,double *za,
                                int *la,int *ma,int *na,float *aexps,float *acoef,
                                double *xb,double *yb,double *zb,
                                int *lb,int *mb,int *nb,float *bexps,float *bcoef,
                                int n_combi, int *start_contr, int *end_contr, double *eri);
