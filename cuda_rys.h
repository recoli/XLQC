/*****************************************************************************
 This file is part of the XLQC program.                                      
 Copyright (C) 2015 Xin Li <lixin.reco@gmail.com>                            
                                                                           
 Filename:  cuda_rys.h                                                      
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAXROOTS 7
__device__ int cuda_fact(int n);
__device__ int cuda_binomial(int a, int b);
__device__ void cuda_Roots(int n, double X, double roots[], double weights[]);
__device__ void cuda_Root123(int n, double X, double roots[], double weights[]);
__device__ void cuda_Root4(double X, double roots[], double weights[]);
__device__ void cuda_Root5(double X, double roots[], double weights[]);
__device__ void cuda_Root6(int n,double X, double roots[], double weights[]);
__device__ double cuda_Int1d(double t,int i,int j,int k, int l,
				double xi,double xj, double xk,double xl,
				double alphai,double alphaj,double alphak,double alphal,
				double G[][MAXROOTS]);
__device__ double cuda_rys_coulomb_repulsion(double xa,double ya,double za,double norma,
				int la,int ma,int na,double alphaa,
				double xb,double yb,double zb,double normb,
				int lb,int mb,int nb,double alphab,
				double xc,double yc,double zc,double normc,
				int lc,int mc,int nc,double alphac,
				double xd,double yd,double zd,double normd,
				int ld,int md,int nd,double alphad);
/*
__global__ void cuda_rys_eri(double *xa,double *ya,double *za,double *norma,
				int *la,int *ma,int *na,double *alphaa,double *acoef,
				double *xb,double *yb,double *zb,double *normb,
				int *lb,int *mb,int *nb,double *alphab,double *bcoef,
				double *xc,double *yc,double *zc,double *normc,
				int *lc,int *mc,int *nc,double *alphac,double *ccoef,
				double *xd,double *yd,double *zd,double *normd,
				int *ld,int *md,int *nd,double *alphad,double *dcoef,
				int n_contr_ints, int *start_contr, int *end_contr, double *eri);
*/

__global__ void cuda_rys_eri_2d(double *xa,double *ya,double *za,
				int *la,int *ma,int *na,double *aexps,double *acoef,
				double *xb,double *yb,double *zb,
				int *lb,int *mb,int *nb,double *bexps,double *bcoef,
				int n_combi, int *start_contr, int *end_contr, double *eri);
