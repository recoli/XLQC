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

__global__ void cuda_rys_eri(double xa,double ya,double za,double norma,
			 int la,int ma,int na,double alphaa,
			 double xb,double yb,double zb,double normb,
			 int lb,int mb,int nb,double alphab,
			 double xc,double yc,double zc,double normc,
			 int lc,int mc,int nc,double alphac,
			 double xd,double yd,double zd,double normd,
			 int ld,int md,int nd,double alphad,
			 double *p_eri);
