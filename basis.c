#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define CART_DIM 3
#define MAX_STR_LEN 256

//=======================================
// allocate memory with failure checking
//=======================================
void* my_malloc(size_t bytes) 
{
    void* ptr = malloc(bytes);
    if(NULL == ptr) 
    {
        fprintf(stderr, "Error: could not allocate memory!\n");
        exit(1);
    } 
    return ptr;
}

//=============================
// combination index
//=============================
int ij2intindex(int i, int j)
{
	if (i < j) { return ij2intindex(j, i); }
	return i * (i + 1) / 2 + j;
}

//===================================
// get nuclear charge of an element
//===================================
int get_nuc_chg(char *element)
{
	char *periodic[] = {"X", 
	"H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", 
	"Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", 
	"Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
	"Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", 
	"Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", 
	"Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", 
	"Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", 
	"Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", 
	"Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra"};

	int z;
	for (z = 1; z <= 88; ++ z)
	{
		if (0 == strcmp(element, periodic[z]))
		{
			return z;
		}
	}

	fprintf(stderr, "Error in get_nuc_chg: cannot find element %s!\n", element);
	exit(1);
}

//=============================
// get number of atoms 
//=============================
int get_natoms(void)
{
	FILE *f_geom;
	f_geom = fopen("geom.dat","r");
	if (NULL == f_geom)
	{
		fprintf(stderr, "Error: cannot open file geom.dat!\n");
		exit(1);
	}

	char line[MAX_STR_LEN];
	if (fgets(line, MAX_STR_LEN, f_geom) != NULL)
	{
		int natoms;
		sscanf(line, "%d", &natoms);
		return natoms;
	}
	else
	{
		fprintf(stderr, "Error in the first line of geom.dat!\n");
		exit(1);
	}
}

//================================
// read geometry
//================================
void read_geom(double **atom_pos, int *atom_nuc_chg, char **atom_name)
{
	FILE *f_geom;
	f_geom = fopen("geom.dat","r");
	if (NULL == f_geom)
	{
		fprintf(stderr, "Error: cannot open file geom.dat!\n");
		exit(1);
	}

	int natoms = 0;
	char line[MAX_STR_LEN];
	if (fgets(line, MAX_STR_LEN, f_geom) != NULL)
	{
		sscanf(line, "%d", &natoms);
	}

	int iatom;
	for (iatom = 0; iatom < natoms; ++ iatom)
	{
		if (fgets(line, MAX_STR_LEN, f_geom) != NULL)
		{
			sscanf(line, "%s%lf%lf%lf", atom_name[iatom], 
					&atom_pos[iatom][0], &atom_pos[iatom][1], &atom_pos[iatom][2]);

			atom_nuc_chg[iatom] = get_nuc_chg(atom_name[iatom]);
		}
	}
}

//======================================
// parse basis set
// get number of basis functions
//======================================
void parse_basis(int natoms, char **atom_name, int *atom_nuc_chg, int *p_nbasis)
{
	FILE *f_basis;
	f_basis = fopen("basis.dat","r");
	if (NULL == f_basis)
	{
		fprintf(stderr, "Error: cannot open file basis.dat!\n");
		exit(1);
	}

	char line[MAX_STR_LEN], cart_type[5], elem_name[5];
	int  int_num, nprims;
	double  dbl_num;

	// count number of elements
	int ielem = 0;
	// nuclear charge and number of basis functions for each element
	int *elem_nuc_chg = (int *)my_malloc(sizeof(int) * 1);
	int *elem_n_basis = (int *)my_malloc(sizeof(int) * 1);

	// loop over elements
	while (1)
	{
		// reallocate elem_nuc_chg and elem_n_basis 
		// if there are more than one elements
		if (ielem > 0)
		{
			int *tmp;

			tmp = realloc(elem_nuc_chg, sizeof(int) * (ielem + 1));
			if (tmp != NULL) { elem_nuc_chg = tmp; }

			tmp = realloc(elem_n_basis, sizeof(int) * (ielem + 1));
			if (tmp != NULL) { elem_n_basis = tmp; }
		}

		// get element name and initialize number of basis functions
		if (fgets(line, MAX_STR_LEN, f_basis) != NULL)
		{
			sscanf(line, "%s%d", elem_name, &int_num);
			elem_nuc_chg[ielem] = get_nuc_chg(elem_name);

			elem_n_basis[ielem] = 0.0;

			if (int_num != 0) 
			{ 
				fprintf(stderr, "Error: int_num=%d is not zero!\n", int_num); 
				exit(1);
			}
		}
		else
		{
			break;
		}

		// write basis functions for this element to a separate file
		char elem_file[MAX_STR_LEN];
		sprintf(elem_file, "basis_%s.dat", elem_name);

		FILE *f_elem;
		f_elem = fopen(elem_file,"w");
		if (NULL == f_elem)
		{
			fprintf(stderr, "Error: cannot write to file %s!\n", elem_file);
			exit(1);
		}

		fprintf(f_elem, "%s     %d\n", elem_name, int_num);

		// loop over contraction blocks
		while (1)
		{
			if (fgets(line, MAX_STR_LEN, f_basis) != NULL)
			{
				sscanf(line, "%s%d%lf", cart_type, &nprims, &dbl_num);

				if (0 == strcmp(cart_type, "****")) { break; }

				if (dbl_num != 1.00) 
				{ 
					fprintf(stderr, "Error: dbl_num=%.2f is not 1.00!\n", dbl_num); 
					exit(1);
				}

				// count number of basis functions for this element
				if      (0 == strcmp(cart_type, "S"))  { elem_n_basis[ielem] += 1; }
				else if (0 == strcmp(cart_type, "SP")) { elem_n_basis[ielem] += 4; }
				else if (0 == strcmp(cart_type, "P"))  { elem_n_basis[ielem] += 3; }
				else if (0 == strcmp(cart_type, "D"))  { elem_n_basis[ielem] += 6; }
				else 
				{ 
					fprintf(stderr, "Error: Cartesian type %s not supported!\n", cart_type); 
					exit(1); 
				}

				fprintf(f_elem, "%s   %d   %.2f\n", cart_type, nprims, dbl_num);

				int iprim;
				for (iprim = 0; iprim < nprims; ++ iprim)
				{
					if (fgets(line, MAX_STR_LEN, f_basis) != NULL)
					{
						fprintf(f_elem, "%s", line);
					}
				}
			}
		}

		fclose(f_elem);
		printf("Element %s   Nbasis %d\n", elem_name, elem_n_basis[ielem]);

		// update counter for elements
		++ ielem;
	}

	// total number of elements
	int nelems = ielem;

	fclose(f_basis);


	// initialize number of basis functions for the whole system
	*p_nbasis = 0;

	// write basis functions for each atom into basis_all.dat
	FILE *f_basis_all;
	f_basis_all = fopen("basis_all.dat", "w");
	if (NULL == f_basis_all)
	{
		fprintf(stderr, "Error: cannot write to file basis_all.dat!\n");
		exit(1);
	}

	int iatom;
	for (iatom = 0; iatom < natoms; ++ iatom)
	{
		// find the correct element for this atom
		// and count number of basis functions
		int found = 0;
		for (ielem = 0; ielem < nelems; ++ ielem)
		{
			if (atom_nuc_chg[iatom] == elem_nuc_chg[ielem])
			{
				*p_nbasis += elem_n_basis[ielem];
				found = 1;
				break;
			}
		}

		if (0 == found)
		{
			fprintf(stderr, "Error: no basis function for atom %d (%s)!\n", iatom+1, atom_name[iatom]);
			exit(1);
		}

		// read basis set for this atom
		// and write to basis_all.dat
		char atom_file[MAX_STR_LEN];
		sprintf(atom_file, "basis_%s.dat", atom_name[iatom]);

		FILE *f_atom;
		f_atom = fopen(atom_file,"r");
		if (NULL == f_atom)
		{
			fprintf(stderr, "Error: cannot open file %s!\n", atom_file);
			exit(1);
		}

		while (fgets(line, MAX_STR_LEN, f_atom) != NULL)
		{
			fprintf(f_basis_all, "%s", line);
		}
		fclose(f_atom);

		fprintf(f_basis_all, "****\n");
	}

	fclose(f_basis_all);


	free(elem_nuc_chg);
	free(elem_n_basis);
}

//==================================================
// read the full basis set created by parse_basis
//==================================================
void read_basis(int natoms, double **atom_pos, int nbasis, double **expon, double **coef, 
				int *nprims, int **lmn, double **xbas)
{
	FILE *f_basis_all;
	f_basis_all = fopen("basis_all.dat","r");
	if (NULL == f_basis_all)
	{
		fprintf(stderr, "Error: cannot open file basis_all.dat!\n");
		exit(1);
	}

	char line[MAX_STR_LEN], cart_type[5];

	// counter for basis functions and atoms
	int ibasis = 0;
	int iatom = 0;

	// loop over elements
	while (1)
	{
		// first line; H   0
		if (NULL == fgets(line, MAX_STR_LEN, f_basis_all))
		{
			break;
		}

		// loop over blocks
		while (1)
		{
			if (fgets(line, MAX_STR_LEN, f_basis_all) != NULL)
			{
				double dbl_num;
				sscanf(line, "%s%d%lf", cart_type, &nprims[ibasis], &dbl_num);

				if (0 == strcmp(cart_type, "****")) { break; }

				int iprim;
				for (iprim = 0; iprim < nprims[ibasis]; ++ iprim)
				{
					if (fgets(line, MAX_STR_LEN, f_basis_all) != NULL)
					{
						// S
						if (0 == strcmp(cart_type, "S"))
						{
							double expon_s, coef_s;
							sscanf(line, "%lf%lf", &expon_s, &coef_s);

							// at the beginning, allocate memories for expon and coef
							// also assign xbas (center of basis funct) and Cartesian lmn
							if (0 == iprim)
							{
								expon[ibasis] = (double *)my_malloc(sizeof(double) * nprims[ibasis]);
								coef[ibasis]  = (double *)my_malloc(sizeof(double) * nprims[ibasis]);

								xbas[ibasis][0] = atom_pos[iatom][0];
								xbas[ibasis][1] = atom_pos[iatom][1];
								xbas[ibasis][2] = atom_pos[iatom][2];

								lmn[ibasis][0] = 0;
								lmn[ibasis][1] = 0;
								lmn[ibasis][2] = 0;
							}

							expon[ibasis][iprim] = expon_s;
							coef[ibasis][iprim]  = coef_s;
						}
						
						// SP
						else if (0 == strcmp(cart_type, "SP"))
						{
							double expon_sp, coef_s, coef_p;
							sscanf(line, "%lf%lf%lf", &expon_sp, &coef_s, &coef_p);

							// S
							if (0 == iprim)
							{
								expon[ibasis] = (double *)my_malloc(sizeof(double) * nprims[ibasis]);
								coef[ibasis]  = (double *)my_malloc(sizeof(double) * nprims[ibasis]);

								xbas[ibasis][0] = atom_pos[iatom][0];
								xbas[ibasis][1] = atom_pos[iatom][1];
								xbas[ibasis][2] = atom_pos[iatom][2];

								lmn[ibasis][0] = 0;
								lmn[ibasis][1] = 0;
								lmn[ibasis][2] = 0;
							}

							expon[ibasis][iprim] = expon_sp;
							coef[ibasis][iprim]  = coef_s;

							int ii, kk;
							int p_lmn[3][3] = {{1,0,0}, {0,1,0}, {0,0,1}};

							// Px, Py, Pz
							if (0 == iprim)
							{
								for (ii = 1; ii <= 3; ++ ii)
								{
									nprims[ibasis + ii] = nprims[ibasis];
									expon[ibasis + ii] = (double *)my_malloc(sizeof(double) * nprims[ibasis]);
									coef[ibasis + ii]  = (double *)my_malloc(sizeof(double) * nprims[ibasis]);

									for (kk = 0; kk < CART_DIM; ++ kk)
									{
										xbas[ibasis + ii][kk] = atom_pos[iatom][kk];
										lmn[ibasis + ii][kk] = p_lmn[ii - 1][kk]; // NOTE: ii-1, not ii!
									}
								}
							}

							for (ii = 1; ii <= 3; ++ ii)
							{
								expon[ibasis + ii][iprim] = expon_sp;
								coef[ibasis + ii][iprim] = coef_p;
							}
						}
						
						// P
						else if (0 == strcmp(cart_type, "P"))
						{
							double expon_p, coef_p;
							sscanf(line, "%lf%lf", &expon_p, &coef_p);

							int ii, kk;
							int p_lmn[3][3] = {{1,0,0}, {0,1,0}, {0,0,1}};

							// Px, Py, Pz
							if (0 == iprim)
							{
								for (ii = 0; ii <= 2; ++ ii)
								{
									nprims[ibasis + ii] = nprims[ibasis];
									expon[ibasis + ii] = (double *)my_malloc(sizeof(double) * nprims[ibasis]);
									coef[ibasis + ii]  = (double *)my_malloc(sizeof(double) * nprims[ibasis]);

									for (kk = 0; kk < CART_DIM; ++ kk)
									{
										xbas[ibasis + ii][kk] = atom_pos[iatom][kk];
										lmn[ibasis + ii][kk] = p_lmn[ii][kk];
									}
								}
							}

							for (ii = 0; ii <= 2; ++ ii)
							{
								expon[ibasis + ii][iprim] = expon_p;
								coef[ibasis + ii][iprim] = coef_p;
							}
						}

						// D
						else if (0 == strcmp(cart_type, "D"))
						{
							double expon_d, coef_d;
							sscanf(line, "%lf%lf", &expon_d, &coef_d);

							int ii, kk;
							int d_lmn[6][3] = {{2,0,0}, {1,1,0}, {1,0,1},
											   {0,2,0}, {0,1,1}, {0,0,2}};

							// Dx2, Dxy, Dxz, Dy2, Dyz, Dz2
							if (0 == iprim)
							{
								for (ii = 0; ii <= 5; ++ ii)
								{
									nprims[ibasis + ii] = nprims[ibasis];
									expon[ibasis + ii] = (double *)my_malloc(sizeof(double) * nprims[ibasis]);
									coef[ibasis + ii]  = (double *)my_malloc(sizeof(double) * nprims[ibasis]);

									for (kk = 0; kk < CART_DIM; ++ kk)
									{
										xbas[ibasis + ii][kk] = atom_pos[iatom][kk];
										lmn[ibasis + ii][kk] = d_lmn[ii][kk];
									}
								}
							}

							for (ii = 0; ii <= 5; ++ ii)
							{
								expon[ibasis + ii][iprim] = expon_d;
								coef[ibasis + ii][iprim] = coef_d;
							}
						}
					}
				}

				if      (0 == strcmp(cart_type, "S"))  { ibasis += 1; }
				else if (0 == strcmp(cart_type, "SP")) { ibasis += 4; }
				else if (0 == strcmp(cart_type, "P"))  { ibasis += 3; }
				else if (0 == strcmp(cart_type, "D"))  { ibasis += 6; }
			}
		}

		++ iatom;
	}

	// check number of atoms and basis functions
	if (iatom != natoms)
	{
		fprintf(stderr, "Error: iatom=%d, natoms=%d\n", iatom, natoms);
	}
	if (ibasis != nbasis)
	{
		fprintf(stderr, "Error: ibasis=%d, nbasis=%d\n", ibasis, nbasis);
	}

	fclose(f_basis_all);
}
