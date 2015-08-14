/**********************************************************************
 * chgp.c  C implementation of two-electron integral code in hgp.py
 *
 * Implementation of Head-Gordon & Pople's scheme for electron repulsion
 *  integrals (ref), which, in turn, derives from Saika and Obarra's scheme.
 *
 * Routines:
 * hrr performs the horizontal recursion relationships
 * vrr performs the vertical recursion relationship

 This program is part of the PyQuante quantum chemistry program suite.

 Copyright (c) 2004, Richard P. Muller. All Rights Reserved. 

 PyQuante version 1.2 and later is covered by the modified BSD
 license. Please see the file LICENSE that is part of this
 distribution. 

 This file has been modified by Xin Li on 2015-08-14.
 1. Removed "static" from all functions
 *
 **********************************************************************/

//#include "Python.h"
//===== Xin Li: added cints.h ==========
#include "cints.h"
#include "chgp.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*======== commented by Xin Li, 2015-08-14 ===============
// Not required for MSVC since the code is included below
#if defined(_WIN32) && !defined(_MSC_VER)
double lgamma(double x);
#endif
======== end of comment ===============*/

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define ITMAX 100
#define EPS 3.0e-7
#define FPMIN 1.0e-30
#define SMALL 0.00000001

#define MAXMTOT 100
double Fgterms[MAXMTOT];

/* MAXAM is one more than the maximum value of an AM (l,m,n) */
#define MAXAM 7
#define MAXTERMS MAXAM*MAXAM*MAXAM*MAXAM*MAXAM*MAXAM*MAXMTOT
double vrr_terms[MAXTERMS];

// lgamma not included in ANSI standard and so not available in MSVC
#if defined(_MSC_VER)
double lgamma(double z) {
    double c[7];
    double x,y ,tmp, ser, v;
    int i;

    if (z<=0) return 0;

    c[0]=2.5066282746310005;
    c[1]=76.18009172947146;
    c[2]=-86.50532032941677;
    c[3]=24.01409824083091;
    c[4]=-1.231739572450155;
    c[5]=0.1208650973866179e-2;
    c[6]=-0.5395239384953e-5;
   
    x   = z;
    y   = x;
    tmp = x+5.5;
    tmp = (x+0.5)*log(tmp)-tmp;
    ser = 1.000000000190015;
    for (i=1; i<7; i++) {
        y   += 1.0;
        ser += c[i]/y;
        }
    v = tmp+log(c[0]*ser/x);
    return v;
    }
#endif

double contr_hrr(int lena, double xa, double ya, double za, double *anorms,
		 int la, int ma, int na, double *aexps, double *acoefs,
		 int lenb, double xb, double yb, double zb, double *bnorms,
		 int lb, int mb, int nb, double *bexps, double *bcoefs,
		 int lenc, double xc, double yc, double zc, double *cnorms,
		 int lc, int mc, int nc, double *cexps, double *ccoefs,
		 int lend, double xd, double yd, double zd, double *dnorms,
		 int ld, int md, int nd, double *dexps, double *dcoefs){
  if (lb > 0) {
    return contr_hrr(lena,xa,ya,za,anorms,la+1,ma,na,aexps,acoefs,
		     lenb,xb,yb,zb,bnorms,lb-1,mb,nb,bexps,bcoefs,
		     lenc,xc,yc,zc,cnorms,lc,mc,nc,cexps,ccoefs,
		     lend,xd,yd,zd,dnorms,ld,md,nd,dexps,dcoefs)
      + (xa-xb)*contr_hrr(lena,xa,ya,za,anorms,la,ma,na,aexps,acoefs,
			  lenb,xb,yb,zb,bnorms,lb-1,mb,nb,bexps,bcoefs,
			  lenc,xc,yc,zc,cnorms,lc,mc,nc,cexps,ccoefs,
			  lend,xd,yd,zd,dnorms,ld,md,nd,dexps,dcoefs);
  }else if (mb > 0){
    return contr_hrr(lena,xa,ya,za,anorms,la,ma+1,na,aexps,acoefs,
		     lenb,xb,yb,zb,bnorms,lb,mb-1,nb,bexps,bcoefs,
		     lenc,xc,yc,zc,cnorms,lc,mc,nc,cexps,ccoefs,
		     lend,xd,yd,zd,dnorms,ld,md,nd,dexps,dcoefs)
      + (ya-yb)*contr_hrr(lena,xa,ya,za,anorms,la,ma,na,aexps,acoefs,
			  lenb,xb,yb,zb,bnorms,lb,mb-1,nb,bexps,bcoefs,
			  lenc,xc,yc,zc,cnorms,lc,mc,nc,cexps,ccoefs,
			  lend,xd,yd,zd,dnorms,ld,md,nd,dexps,dcoefs);
  }else if (nb > 0){
    return contr_hrr(lena,xa,ya,za,anorms,la,ma,na+1,aexps,acoefs,
		     lenb,xb,yb,zb,bnorms,lb,mb,nb-1,bexps,bcoefs,
		     lenc,xc,yc,zc,cnorms,lc,mc,nc,cexps,ccoefs,
		     lend,xd,yd,zd,dnorms,ld,md,nd,dexps,dcoefs)
      + (za-zb)*contr_hrr(lena,xa,ya,za,anorms,la,ma,na,aexps,acoefs,
			  lenb,xb,yb,zb,bnorms,lb,mb,nb-1,bexps,bcoefs,
			  lenc,xc,yc,zc,cnorms,lc,mc,nc,cexps,ccoefs,
			  lend,xd,yd,zd,dnorms,ld,md,nd,dexps,dcoefs);
  }else if (ld > 0){
    return contr_hrr(lena,xa,ya,za,anorms,la,ma,na,aexps,acoefs,
		     lenb,xb,yb,zb,bnorms,lb,mb,nb,bexps,bcoefs,
		     lenc,xc,yc,zc,cnorms,lc+1,mc,nc,cexps,ccoefs,
		     lend,xd,yd,zd,dnorms,ld-1,md,nd,dexps,dcoefs)
      + (xc-xd)*contr_hrr(lena,xa,ya,za,anorms,la,ma,na,aexps,acoefs,
			  lenb,xb,yb,zb,bnorms,lb,mb,nb,bexps,bcoefs,
			  lenc,xc,yc,zc,cnorms,lc,mc,nc,cexps,ccoefs,
			  lend,xd,yd,zd,dnorms,ld-1,md,nd,dexps,dcoefs);
  }else if (md > 0){
    return contr_hrr(lena,xa,ya,za,anorms,la,ma,na,aexps,acoefs,
		     lenb,xb,yb,zb,bnorms,lb,mb,nb,bexps,bcoefs,
		     lenc,xc,yc,zc,cnorms,lc,mc+1,nc,cexps,ccoefs,
		     lend,xd,yd,zd,dnorms,ld,md-1,nd,dexps,dcoefs)
      + (yc-yd)*contr_hrr(lena,xa,ya,za,anorms,la,ma,na,aexps,acoefs,
			  lenb,xb,yb,zb,bnorms,lb,mb,nb,bexps,bcoefs,
			  lenc,xc,yc,zc,cnorms,lc,mc,nc,cexps,ccoefs,
			  lend,xd,yd,zd,dnorms,ld,md-1,nd,dexps,dcoefs);
  }else if (nd > 0){
    return contr_hrr(lena,xa,ya,za,anorms,la,ma,na,aexps,acoefs,
		     lenb,xb,yb,zb,bnorms,lb,mb,nb,bexps,bcoefs,
		     lenc,xc,yc,zc,cnorms,lc,mc,nc+1,cexps,ccoefs,
		     lend,xd,yd,zd,dnorms,ld,md,nd-1,dexps,dcoefs)
      + (zc-zd)*contr_hrr(lena,xa,ya,za,anorms,la,ma,na,aexps,acoefs,
			  lenb,xb,yb,zb,bnorms,lb,mb,nb,bexps,bcoefs,
			  lenc,xc,yc,zc,cnorms,lc,mc,nc,cexps,ccoefs,
			  lend,xd,yd,zd,dnorms,ld,md,nd-1,dexps,dcoefs);
  }
  return contr_vrr(lena,xa,ya,za,anorms,la,ma,na,aexps,acoefs,
		   lenb,xb,yb,zb,bnorms,bexps,bcoefs,
		   lenc,xc,yc,zc,cnorms,lc,mc,nc,cexps,ccoefs,
		   lend,xd,yd,zd,dnorms,dexps,dcoefs);
}

double contr_vrr(int lena, double xa, double ya, double za,
			double *anorms, int la, int ma, int na,
			double *aexps, double *acoefs,
			int lenb, double xb, double yb, double zb,
			double *bnorms, double *bexps, double *bcoefs,
			int lenc, double xc, double yc, double zc,
			double *cnorms, int lc, int mc, int nc,
			double *cexps, double *ccoefs,
			int lend, double xd, double yd, double zd,
			double *dnorms, double *dexps, double *dcoefs){
  int i,j,k,l;
  double val=0.;
  for (i=0; i<lena; i++)
    for (j=0; j<lenb; j++)
      for (k=0; k<lenc; k++)
	for (l=0; l<lend; l++)
	  val += acoefs[i]*bcoefs[j]*ccoefs[k]*dcoefs[l]*
	    vrr(xa,ya,za,anorms[i],la,ma,na,aexps[i],
		xb,yb,zb,bnorms[j],bexps[j],
		xc,yc,zc,cnorms[k],lc,mc,nc,cexps[k],
		xd,yd,zd,dnorms[l],dexps[l],0);
  return val;
}


double hrr(double xa, double ya, double za, double norma,
	   int la, int ma, int na, double alphaa,
	   double xb, double yb, double zb, double normb,
	   int lb, int mb, int nb, double alphab,
	   double xc, double yc, double zc, double normc,
	   int lc, int mc, int nc, double alphac,
	   double xd, double yd, double zd, double normd,
	   int ld, int md, int nd, double alphad){
  if (lb > 0) {
    return hrr(xa,ya,za,norma,la+1,ma,na,alphaa,
	       xb,yb,zb,normb,lb-1,mb,nb,alphab,
	       xc,yc,zc,normc,lc,mc,nc,alphac,
	       xd,yd,zd,normd,ld,md,nd,alphad)
      + (xa-xb)*hrr(xa,ya,za,norma,la,ma,na,alphaa,
		    xb,yb,zb,normb,lb-1,mb,nb,alphab,
		    xc,yc,zc,normc,lc,mc,nc,alphac,
		    xd,yd,zd,normd,ld,md,nd,alphad);
  }else if (mb > 0){
    return hrr(xa,ya,za,norma,la,ma+1,na,alphaa,
	       xb,yb,zb,normb,lb,mb-1,nb,alphab,
	       xc,yc,zc,normc,lc,mc,nc,alphac,
	       xd,yd,zd,normd,ld,md,nd,alphad)
      + (ya-yb)*hrr(xa,ya,za,norma,la,ma,na,alphaa,
		    xb,yb,zb,normb,lb,mb-1,nb,alphab,
		    xc,yc,zc,normc,lc,mc,nc,alphac,
		    xd,yd,zd,normd,ld,md,nd,alphad);
  }else if (nb > 0){
    return hrr(xa,ya,za,norma,la,ma,na+1,alphaa,
	       xb,yb,zb,normb,lb,mb,nb-1,alphab,
	       xc,yc,zc,normc,lc,mc,nc,alphac,
	       xd,yd,zd,normd,ld,md,nd,alphad)
      + (za-zb)*hrr(xa,ya,za,norma,la,ma,na,alphaa,
		    xb,yb,zb,normb,lb,mb,nb-1,alphab,
		    xc,yc,zc,normc,lc,mc,nc,alphac,
		    xd,yd,zd,normd,ld,md,nd,alphad);
  }else if (ld > 0){
    return hrr(xa,ya,za,norma,la,ma,na,alphaa,
	       xb,yb,zb,normb,lb,mb,nb,alphab,
	       xc,yc,zc,normc,lc+1,mc,nc,alphac,
	       xd,yd,zd,normd,ld-1,md,nd,alphad)
      + (xc-xd)*hrr(xa,ya,za,norma,la,ma,na,alphaa,
		    xb,yb,zb,normb,lb,mb,nb,alphab,
		    xc,yc,zc,normc,lc,mc,nc,alphac,
		    xd,yd,zd,normd,ld-1,md,nd,alphad);
  }else if (md > 0){
    return hrr(xa,ya,za,norma,la,ma,na,alphaa,
	       xb,yb,zb,normb,lb,mb,nb,alphab,
	       xc,yc,zc,normc,lc,mc+1,nc,alphac,
	       xd,yd,zd,normd,ld,md-1,nd,alphad)
      + (yc-yd)*hrr(xa,ya,za,norma,la,ma,na,alphaa,
		    xb,yb,zb,normb,lb,mb,nb,alphab,
		    xc,yc,zc,normc,lc,mc,nc,alphac,
		    xd,yd,zd,normd,ld,md-1,nd,alphad);
  }else if (nd > 0){
    return hrr(xa,ya,za,norma,la,ma,na,alphaa,
	       xb,yb,zb,normb,lb,mb,nb,alphab,
	       xc,yc,zc,normc,lc,mc,nc+1,alphac,
	       xd,yd,zd,normd,ld,md,nd-1,alphad)
      + (zc-zd)*hrr(xa,ya,za,norma,la,ma,na,alphaa,
		    xb,yb,zb,normb,lb,mb,nb,alphab,
		    xc,yc,zc,normc,lc,mc,nc,alphac,
		    xd,yd,zd,normd,ld,md,nd-1,alphad);
  }
  /* Implicit else: */
  /* When we expand hrr to handle contracted functions as well, */
  /* this routine is expanded to be a sum over all of the primitives. */
  return vrr(xa,ya,za,norma,la,ma,na,alphaa,
	     xb,yb,zb,normb,alphab,
	     xc,yc,zc,normc,lc,mc,nc,alphac,
	     xd,yd,zd,normd,alphad,0);
}

double vrr(double xa, double ya, double za, double norma,
	   int la, int ma, int na, double alphaa,
	   double xb, double yb, double zb, double normb, double alphab,
	   double xc, double yc, double zc, double normc,
	   int lc, int mc, int nc, double alphac,
	   double xd, double yd, double zd, double normd, double alphad,
	   int m){

  double px,py,pz,qx,qy,qz,zeta,eta,wx,wy,wz,rab2,rcd2,Kcd,rpq2,T,Kab,val;
  //double *vrr_terms;

  int i,j,k,q,r,s,im,mtot;

  if ((la > MAXAM) || (ma > MAXAM) || (na > MAXAM) || (lc > MAXAM) ||
      (mc > MAXAM) || (nc > MAXAM)) {
    printf("vrr: MAXAM set to %d\n",MAXAM);
    printf("     current a.m. values are (%d,%d,%d),(%d,%d,%d)\n",la,ma,na,lc,mc,nc);
  }

  px = product_center_1D(alphaa,xa,alphab,xb);
  py = product_center_1D(alphaa,ya,alphab,yb);
  pz = product_center_1D(alphaa,za,alphab,zb);

  qx = product_center_1D(alphac,xc,alphad,xd);
  qy = product_center_1D(alphac,yc,alphad,yd);
  qz = product_center_1D(alphac,zc,alphad,zd);

  zeta = alphaa+alphab;
  eta = alphac+alphad;

  wx = product_center_1D(zeta,px,eta,qx);
  wy = product_center_1D(zeta,py,eta,qy);
  wz = product_center_1D(zeta,pz,eta,qz);

  rab2 = dist2(xa,ya,za,xb,yb,zb);
  Kab = sqrt(2.)*pow(M_PI,1.25)/(alphaa+alphab)
    *exp(-alphaa*alphab/(alphaa+alphab)*rab2);
  rcd2 = dist2(xc,yc,zc,xd,yd,zd);
  Kcd = sqrt(2.)*pow(M_PI,1.25)/(alphac+alphad)
    *exp(-alphac*alphad/(alphac+alphad)*rcd2);
  rpq2 = dist2(px,py,pz,qx,qy,qz);
  T = zeta*eta/(zeta+eta)*rpq2;

  mtot = la+ma+na+lc+mc+nc+m;

  if (la*ma*na*lc*mc*nc*mtot>pow(MAXAM,6)*MAXMTOT) {
    printf("Buffer not large enough in vrr\n");
    printf(" increase MAXAM or MAXMTOT\n");
    exit(1);
  }

  if (mtot > MAXMTOT) printf("Error: MAXMTOT=%d, needs to be > %d\n",MAXMTOT,mtot);
  Fgterms[mtot] = Fgamma(mtot,T);

  for (im=mtot-1; im>=0; im--)
    Fgterms[im]=(2.*T*Fgterms[im+1]+exp(-T))/(2.*im+1);

  for (im=0; im<mtot+1; im++)
    vrr_terms[iindex(0,0,0,0,0,0,im)] = 
      norma*normb*normc*normd*Kab*Kcd/sqrt(zeta+eta)*Fgterms[im];

  for (i=0; i<la; i++){
    for (im=0; im<mtot-i; im++) {
      vrr_terms[iindex(i+1,0,0, 0,0,0, im)] = 
	(px-xa)*vrr_terms[iindex(i,0,0, 0,0,0, im)]
	+ (wx-px)*vrr_terms[iindex(i,0,0, 0,0,0, im+1)];
      
      if (i>0)
	vrr_terms[iindex(i+1,0,0, 0,0,0, im)] += 
	  i/2./zeta*( vrr_terms[iindex(i-1,0,0, 0,0,0, im)]
		      - eta/(zeta+eta)*vrr_terms[iindex(i-1,0,0, 0,0,0, im+1)]);
    }
  }  


  for (j=0; j<ma; j++){
    for (i=0; i<la+1; i++){
      for (im=0; im<mtot-i-j; im++){
	vrr_terms[iindex(i,j+1,0, 0,0,0, im)] = 
	  (py-ya)*vrr_terms[iindex(i,j,0, 0,0,0, im)]
	  + (wy-py)*vrr_terms[iindex(i,j,0, 0,0,0, im+1)];

	if (j>0)
	  vrr_terms[iindex(i,j+1,0, 0,0,0, im)] +=
	    j/2./zeta*(vrr_terms[iindex(i,j-1,0, 0,0,0, im)]
		       - eta/(zeta+eta)
		       *vrr_terms[iindex(i,j-1,0, 0,0,0, im+1)]);
      }
    }
  }

  for (k=0; k<na; k++){
    for (j=0; j<ma+1; j++){
      for (i=0; i<la+1; i++){
	for (im=0; im<mtot-i-j-k; im++){
	  vrr_terms[iindex(i,j,k+1, 0,0,0, im)] = 
	    (pz-za)*vrr_terms[iindex(i,j,k, 0,0,0, im)]
	    + (wz-pz)*vrr_terms[iindex(i,j,k, 0,0,0, im+1)];
	  if (k>0)
	    vrr_terms[iindex(i,j,k+1, 0,0,0, im)] += 
	      k/2./zeta*(vrr_terms[iindex(i,j,k-1, 0,0,0, im)]
			 - eta/(zeta+eta)
			 *vrr_terms[iindex(i,j,k-1, 0,0,0, im+1)]);
	}
      }
    }
  }

  for (q=0; q<lc; q++){
    for (k=0; k<na+1; k++){
      for (j=0; j<ma+1; j++){
	for (i=0; i<la+1; i++){
	  for (im=0; im<mtot-i-j-k-q; im++){
	    vrr_terms[iindex(i,j,k, q+1,0,0, im)] = 
	      (qx-xc)*vrr_terms[iindex(i,j,k, q,0,0, im)]
	      + (wx-qx)*vrr_terms[iindex(i,j,k, q,0,0, im+1)];
	    if (q>0)
	      vrr_terms[iindex(i,j,k, q+1,0,0, im)] += 
		q/2./eta*(vrr_terms[iindex(i,j,k, q-1,0,0, im)]
			  - zeta/(zeta+eta)
			  *vrr_terms[iindex(i,j,k, q-1,0,0, im+1)]);
	    if (i>0)
	      vrr_terms[iindex(i,j,k, q+1,0,0, im)] += 
		i/2./(zeta+eta)*vrr_terms[iindex(i-1,j,k, q,0,0, im+1)];
	  }
	}
      }
    }
  }

  for (r=0; r<mc; r++){
    for (q=0; q<lc+1; q++){
      for (k=0; k<na+1; k++){
	for (j=0; j<ma+1; j++){
	  for (i=0; i<la+1; i++){
	    for (im=0; im<mtot-i-j-k-q-r; im++){
	      vrr_terms[iindex(i,j,k, q,r+1,0, im)] = 
		(qy-yc)*vrr_terms[iindex(i,j,k, q,r,0, im)]
		+ (wy-qy)*vrr_terms[iindex(i,j,k, q,r,0, im+1)];
	      if (r>0)
		vrr_terms[iindex(i,j,k, q,r+1,0, im)] += 
		  r/2./eta*(vrr_terms[iindex(i,j,k, q,r-1,0, im)]
			    - zeta/(zeta+eta)
			    *vrr_terms[iindex(i,j,k, q,r-1,0, im+1)]);
	      if (j>0)
		vrr_terms[iindex(i,j,k, q,r+1,0, im)] += 
		  j/2./(zeta+eta)*vrr_terms[iindex(i,j-1,k,q,r,0,im+1)];
	    }
	  }
	}
      }
    }
  }

  for (s=0; s<nc; s++){
    for (r=0; r<mc+1; r++){
      for (q=0; q<lc+1; q++){
	for (k=0; k<na+1; k++){
	  for (j=0; j<ma+1; j++){
	    for (i=0; i<la+1; i++){
	      for (im=0; im<mtot-i-j-k-q-r-s; im++){
		vrr_terms[iindex(i,j,k,q,r,s+1,im)] = 
		  (qz-zc)*vrr_terms[iindex(i,j,k,q,r,s,im)]
		  + (wz-qz)*vrr_terms[iindex(i,j,k,q,r,s,im+1)];
		if (s>0)
		  vrr_terms[iindex(i,j,k,q,r,s+1,im)] += 
		    s/2./eta*(vrr_terms[iindex(i,j,k,q,r,s-1,im)]
			      - zeta/(zeta+eta)
			      *vrr_terms[iindex(i,j,k,q,r,s-1,im+1)]);
		if (k>0)
		  vrr_terms[iindex(i,j,k,q,r,s+1,im)] += 
		    k/2./(zeta+eta)*vrr_terms[iindex(i,j,k-1,q,r,s,im+1)];
	      }
	    }
	  }
	}
      }
    }
  }
  val = vrr_terms[iindex(la,ma,na,lc,mc,nc,m)];

  return val;

}

/* These iindexing functions are wasteful: they simply allocate an array
   of dimension MAXAM^6*mtot. Once this is working I'll look to 
   actually implement the correct size array la*ma*na*lc*mc*nc*mtot */

int iindex(int la, int ma, int na, int lc, int mc, int nc, int m){
  /* Convert the 7-dimensional indices to a 1d iindex */
  int ival=0;
  ival = la + ma*MAXAM + na*MAXAM*MAXAM + lc*MAXAM*MAXAM*MAXAM +
    nc*MAXAM*MAXAM*MAXAM*MAXAM + mc*MAXAM*MAXAM*MAXAM*MAXAM*MAXAM +
    m*MAXAM*MAXAM*MAXAM*MAXAM*MAXAM*MAXAM;
  return ival;
}

double vrr_recursive(double xa, double ya, double za, double norma,
	   int la, int ma, int na, double alphaa,
	   double xb, double yb, double zb, double normb, double alphab,
	   double xc, double yc, double zc, double normc,
	   int lc, int mc, int nc, double alphac,
	   double xd, double yd, double zd, double normd, double alphad,
	   int m){

  double px,py,pz,qx,qy,qz,zeta,eta,wx,wy,wz,rab2,rcd2,Kcd,rpq2,T,Kab,val;

  px = product_center_1D(alphaa,xa,alphab,xb);
  py = product_center_1D(alphaa,ya,alphab,yb);
  pz = product_center_1D(alphaa,za,alphab,zb);

  qx = product_center_1D(alphac,xc,alphad,xd);
  qy = product_center_1D(alphac,yc,alphad,yd);
  qz = product_center_1D(alphac,zc,alphad,zd);

  zeta = alphaa+alphab;
  eta = alphac+alphad;

  wx = product_center_1D(zeta,px,eta,qx);
  wy = product_center_1D(zeta,py,eta,qy);
  wz = product_center_1D(zeta,pz,eta,qz);


  if (nc > 0){
    val = (qz-zc)*vrr(xa,ya,za,norma,la,ma,na,alphaa,
		      xb,yb,zb,normb,alphab,
		      xc,yc,zc,normc,lc,mc,nc-1,alphac,
		      xd,yd,zd,normd,alphad,m) 
      + (wz-qz)*vrr(xa,ya,za,norma,la,ma,na,alphaa,
		    xb,yb,zb,normb,alphab,
		    xc,yc,zc,normc,lc,mc,nc-1,alphac,
		    xd,yd,zd,normd,alphad,m+1);
    if (nc > 1)
      val += 0.5*(nc-1)/eta*(vrr(xa,ya,za,norma,la,ma,na,alphaa,
				 xb,yb,zb,normb,alphab,
				 xc,yc,zc,normc,lc,mc,nc-2,alphac,
				 xd,yd,zd,normd,alphad,m)
			     -zeta/(zeta+eta)* 
			     vrr(xa,ya,za,norma,la,ma,na,alphaa,
				 xb,yb,zb,normb,alphab,
				 xc,yc,zc,normc,lc,mc,nc-2,alphac,
				 xd,yd,zd,normd,alphad,m+1) );
    if (na > 0)
      val += 0.5*na/(zeta+eta)*vrr(xa,ya,za,norma,la,ma,na-1,alphaa,
				   xb,yb,zb,normb,alphab,
				   xc,yc,zc,normc,lc,mc,nc-1,alphac,
				   xd,yd,zd,normd,alphad,m+1);
    return val;
  }else if (mc > 0){
    val = (qy-yc)*vrr(xa,ya,za,norma,la,ma,na,alphaa,
		      xb,yb,zb,normb,alphab,
		      xc,yc,zc,normc,lc,mc-1,nc,alphac,
		      xd,yd,zd,normd,alphad,m) 
      + (wy-qy)*vrr(xa,ya,za,norma,la,ma,na,alphaa,
		    xb,yb,zb,normb,alphab,
		    xc,yc,zc,normc,lc,mc-1,nc,alphac,
		    xd,yd,zd,normd,alphad,m+1);
    if (mc > 1)
      val += 0.5*(mc-1)/eta*(vrr(xa,ya,za,norma,la,ma,na,alphaa,
				 xb,yb,zb,normb,alphab,
				 xc,yc,zc,normc,lc,mc-2,nc,alphac,
				 xd,yd,zd,normd,alphad,m)
			     -zeta/(zeta+eta)* 
			     vrr(xa,ya,za,norma,la,ma,na,alphaa,
				 xb,yb,zb,normb,alphab,
				 xc,yc,zc,normc,lc,mc-2,nc,alphac,
				 xd,yd,zd,normd,alphad,m+1) );
    if (ma > 0)
      val += 0.5*ma/(zeta+eta)*vrr(xa,ya,za,norma,la,ma-1,na,alphaa,
				   xb,yb,zb,normb,alphab,
				   xc,yc,zc,normc,lc,mc-1,nc,alphac,
				   xd,yd,zd,normd,alphad,m+1);
    return val;
  }else if (lc > 0){
    val = (qx-xc)*vrr(xa,ya,za,norma,la,ma,na,alphaa,
		      xb,yb,zb,normb,alphab,
		      xc,yc,zc,normc,lc-1,mc,nc,alphac,
		      xd,yd,zd,normd,alphad,m) 
      + (wx-qx)*vrr(xa,ya,za,norma,la,ma,na,alphaa,
		    xb,yb,zb,normb,alphab,
		    xc,yc,zc,normc,lc-1,mc,nc,alphac,
		    xd,yd,zd,normd,alphad,m+1);
    if (lc > 1)
      val += 0.5*(lc-1)/eta*(vrr(xa,ya,za,norma,la,ma,na,alphaa,
				 xb,yb,zb,normb,alphab,
				 xc,yc,zc,normc,lc-2,mc,nc,alphac,
				 xd,yd,zd,normd,alphad,m)
			     -zeta/(zeta+eta)* 
			     vrr(xa,ya,za,norma,la,ma,na,alphaa,
				 xb,yb,zb,normb,alphab,
				 xc,yc,zc,normc,lc-2,mc,nc,alphac,
				 xd,yd,zd,normd,alphad,m+1) );
    if (la > 0)
      val += 0.5*la/(zeta+eta)*vrr(xa,ya,za,norma,la-1,ma,na,alphaa,
				   xb,yb,zb,normb,alphab,
				   xc,yc,zc,normc,lc-1,mc,nc,alphac,
				   xd,yd,zd,normd,alphad,m+1);
    return val;
  }else if (na > 0) {
    val = (pz-za)*vrr(xa,ya,za,norma,la,ma,na-1,alphaa,
		      xb,yb,zb,normb,alphab,
		      xc,yc,zc,normc,lc,mc,nc,alphac,
		      xd,yd,zd,normd,alphad,m) 
      + (wz-pz)*vrr(xa,ya,za,norma,la,ma,na-1,alphaa,
		    xb,yb,zb,normb,alphab,
		    xc,yc,zc,normc,lc,mc,nc,alphac,
		    xd,yd,zd,normd,alphad,m+1);
        
    if (na > 1)
      val +=  0.5*(na-1)/zeta*(vrr(xa,ya,za,norma,la,ma,na-2,alphaa,
				   xb,yb,zb,normb,alphab,
				   xc,yc,zc,normc,lc,mc,nc,alphac,
				   xd,yd,zd,normd,alphad,m)
			       -eta/(zeta+eta)* 
			       vrr(xa,ya,za,norma,la,ma,na-2,alphaa,
				   xb,yb,zb,normb,alphab,
				   xc,yc,zc,normc,lc,mc,nc,alphac,
				   xd,yd,zd,normd,alphad,m+1) );
    return val;
  }else if (ma > 0) {
    val = (py-ya)*vrr(xa,ya,za,norma,la,ma-1,na,alphaa,
		      xb,yb,zb,normb,alphab,
		      xc,yc,zc,normc,lc,mc,nc,alphac,
		      xd,yd,zd,normd,alphad,m) 
      + (wy-py)*vrr(xa,ya,za,norma,la,ma-1,na,alphaa,
		    xb,yb,zb,normb,alphab,
		    xc,yc,zc,normc,lc,mc,nc,alphac,
		    xd,yd,zd,normd,alphad,m+1);
    if (ma > 1)
      val +=  0.5*(ma-1)/zeta*(vrr(xa,ya,za,norma,la,ma-2,na,alphaa,
				   xb,yb,zb,normb,alphab,
				   xc,yc,zc,normc,lc,mc,nc,alphac,
				   xd,yd,zd,normd,alphad,m)
			       -eta/(zeta+eta)* 
			       vrr(xa,ya,za,norma,la,ma-2,na,alphaa,
				   xb,yb,zb,normb,alphab,
				   xc,yc,zc,normc,lc,mc,nc,alphac,
				   xd,yd,zd,normd,alphad,m+1) );
    return val;
  }else if (la > 0) {
    val = (px-xa)*vrr(xa,ya,za,norma,la-1,ma,na,alphaa,
		      xb,yb,zb,normb,alphab,
		      xc,yc,zc,normc,lc,mc,nc,alphac,
		      xd,yd,zd,normd,alphad,m) 
      + (wx-px)*vrr(xa,ya,za,norma,la-1,ma,na,alphaa,
		    xb,yb,zb,normb,alphab,
		    xc,yc,zc,normc,lc,mc,nc,alphac,
		    xd,yd,zd,normd,alphad,m+1);
    if (la > 1)
      val +=  0.5*(la-1)/zeta*(vrr(xa,ya,za,norma,la-2,ma,na,alphaa,
				   xb,yb,zb,normb,alphab,
				   xc,yc,zc,normc,lc,mc,nc,alphac,
				   xd,yd,zd,normd,alphad,m)
			       -eta/(zeta+eta)* 
			       vrr(xa,ya,za,norma,la-2,ma,na,alphaa,
				   xb,yb,zb,normb,alphab,
				   xc,yc,zc,normc,lc,mc,nc,alphac,
				   xd,yd,zd,normd,alphad,m+1) );
    return val;
  }
  
  rab2 = dist2(xa,ya,za,xb,yb,zb);
  Kab = sqrt(2.)*pow(M_PI,1.25)/(alphaa+alphab)
    *exp(-alphaa*alphab/(alphaa+alphab)*rab2);
  rcd2 = dist2(xc,yc,zc,xd,yd,zd);
  Kcd = sqrt(2.)*pow(M_PI,1.25)/(alphac+alphad)
    *exp(-alphac*alphad/(alphac+alphad)*rcd2);
  rpq2 = dist2(px,py,pz,qx,qy,qz);
  T = zeta*eta/(zeta+eta)*rpq2;
  val = norma*normb*normc*normd*Kab*Kcd/sqrt(zeta+eta)*Fgamma(m,T);
  return val;
}

/*=========== util funcs removed by Xin Li, 2015-08-14 ===========
 * since these functions are already in cints.h
 *========= end of comment ===========*/

/*======== python wrappers removed by Xin Li, 2015-08-14 ==========*/
