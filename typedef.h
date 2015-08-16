/*****************************************************************************
 This file is part of the XLQC program.                                      
 Copyright (C) 2015 Xin Li <lixin.reco@gmail.com>                            
                                                                           
 Filename:  typedef.h                                                      
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

// Cartesian dimension
#define CART_DIM 3

#define N_S   1
#define N_SP  4
#define N_P   3
#define N_D   6
#define N_F  10

// maximal number of DIIS error matrices
#define MAX_DIIS_DIM 6

// maximal length of string
#define MAX_STR_LEN 256

// PI
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 2.0 * M_PI
#define TWO_PI 6.28318530717959

// 1.0 / sqrt(M_PI)
#define INV_SQRT_PI 0.564189583547756

// CODATA 2014: 1 Hartree = 27.21138602 eV
#define HARTREE2EV 27.21138602

// 1.0 eV = 96.4853363969351 kJ/mol
#define EV2KJMOL 96.4853363969351

// 1.0 Angstrom = 1.88972612456506 Bohr
#define AA2BOHR 1.88972612456506

// Boltzmann constant, in kJ mol^-1 K^-1
#define K_BOLTZ 0.00831446214546895

// vector
typedef struct {
	double x, y, z;
} Vec_R;

// atomic name, position and nuclear charge
typedef struct {
	int num;
	char **name;
	double **pos;
	int *nuc_chg;
} Atom;

// basis set
typedef struct {
	int num;
	double **expon, **coef, **xbas, **norm;
	int *nprims;
	int **lmn;
} Basis;

