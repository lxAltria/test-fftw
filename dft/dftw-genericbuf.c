/*
 * Copyright (c) 2003, 2007-14 Matteo Frigo
 * Copyright (c) 2003, 2007-14 Massachusetts Institute of Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

/* express a twiddle problem in terms of dft + multiplication by
   twiddle factors */

#include "ct.h"
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <unistd.h>

extern bool dmr;
extern R * outChecksum1; // intermediate checksum after first part
extern R * outChecksum2;
extern R * rA;
extern int fft2Radix;
extern int myRadix;
extern int myRank;
extern int pNum;
extern R * outputChecksum1; // checksum for output before the third tranpose
extern R * outputChecksum2;

extern clock_t t1;

R * dftwIPC;
R * interChecksum1;

bool inject2 = false;

#define dftwDelta 1e-2
#define dftwDelta2 1e-2

typedef struct {
     ct_solver super;
     INT batchsz;
} S;

typedef struct {
     plan_dftw super;

     INT r, rs, m, ms, v, vs, mb, me;
     INT batchsz;
     plan *cld;

     triggen *t;
     const S *slv;
} P;

// copied from kernel/trig.c
#if defined(TRIGREAL_IS_LONG_DOUBLE)
#  define COS cosl
#  define SIN sinl
#  define KTRIG(x) (x##L)
#  if defined(HAVE_DECL_SINL) && !HAVE_DECL_SINL
     extern long double sinl(long double x);
#  endif
#  if defined(HAVE_DECL_COSL) && !HAVE_DECL_COSL
     extern long double cosl(long double x);
#  endif
#elif defined(TRIGREAL_IS_QUAD)
#  define COS cosq
#  define SIN sinq
#  define KTRIG(x) (x##Q)
   extern __float128 sinq(__float128 x);
   extern __float128 cosq(__float128 x);
#else
#  define COS cos
#  define SIN sin
#  define KTRIG(x) (x)
#endif

static const trigreal K2PI =
    KTRIG(6.2831853071795864769252867665590057683943388);
#define by2pi(m, n) ((K2PI * (m)) / (n))

/*
 * Improve accuracy by reducing x to range [0..1/8]
 * before multiplication by 2 * PI.
 */

static void real_cexp(INT m, INT n, trigreal *out)
{
     trigreal theta, c, s, t;
     unsigned octant = 0;
     INT quarter_n = n;

     n += n; n += n;
     m += m; m += m;

     if (m < 0) m += n;
     if (m > n - m) { m = n - m; octant |= 4; }
     if (m - quarter_n > 0) { m = m - quarter_n; octant |= 2; }
     if (m > quarter_n - m) { m = quarter_n - m; octant |= 1; }

     theta = by2pi(m, n);
     c = COS(theta); s = SIN(theta);

     if (octant & 1) { t = c; c = s; s = t; }
     if (octant & 2) { t = c; c = -s; s = t; }
     if (octant & 4) { s = -s; }

     out[0] = c; 
     out[1] = s; 
}
// copy done

#define BATCHDIST(r) ((r) + 16)

/**************************************************************/
static void bytwiddle(const P *ego, INT mb, INT me, R *buf, R *rio, R *iio)
{
     INT j, k;
     INT r = ego->r, rs = ego->rs, ms = ego->ms;
     triggen *t = ego->t;

     // printf("processor %d bytwiddle start\n", myRank);
     // fflush(stdout);

     clock_t start, end;
     double eTime;
     int uSleepTime;

     int batchSize = ego->batchsz; 
     bool recalc = true;
     while(recalc)
     {
          R ipcReal = 0, ipcIm = 0;
          memset(interChecksum1, 0, 2*batchSize*sizeof(R));

          R * rA0 = rA;
          R sum00, sum01;
          register R temp0, temp1;
          R * sumPos1;
          // R * sumPos2;

          start = clock();

          register int index;
          for (j = 0; j < r; ++j) {
            sum00 = 0, sum01 = 0;
            sumPos1 = interChecksum1;
            // sumPos2 = interChecksum2;
            for (k = mb; k < me; ++k)
            {
               index = j * rs + k * ms;
               temp0 = rio[index];
               temp1 = iio[index];
               sumPos1[0] += temp0, sumPos1[1] += temp1;
               // sumPos2[0] += cof * temp0, sumPos1[1] += cof * temp1;               
               index = j * 2 + 2 * BATCHDIST(r) * (k - mb) + 0;
               t->rotate(t, j * k, temp0, temp1, &buf[index]);
               sum00 += buf[index], sum01 += buf[index+1];
               sumPos1 += 2;
            }
            ipcReal += rA0[0]*sum00 - rA0[1]*sum01;
            ipcIm += rA0[1]*sum00 + rA0[0]*sum01;
            rA0 += 2;
          }

          end = clock();
          eTime = ((double)(end - start)) / CLOCKS_PER_SEC;
          uSleepTime = (int) (eTime * 3*1e5);
          usleep(uSleepTime);

          // check
          recalc = false;
          sumPos1 = interChecksum1;
          R diff;
          for(k = mb; k<me; k++)
          {
               diff = sumPos1[0] - outChecksum1[2*k];
               if(diff < -dftwDelta || diff > dftwDelta)
               {
                    printf("processor %d memory error in real part of %d-th FFT bytwiddle, sum %.10f cks %.10f diff %.10f\n", myRank, k, sumPos1[0], outChecksum1[2*k], diff);
                    // fflush(stdout);
                    recalc = true;

                    R sum10 = 0;

                    for(int j=0; j<r; j++)
                    {
                         sum10 += (j+1) * rio[j*rs + k*ms];
                    }
                    R diff2 = sum10 - outChecksum2[2*k];
                      
                    // int index =(int) (diff / diff2 - 0.5); // -1 for index
                    int index =(int) (diff2 / diff - 0.5); // -1 for index

                    int pos = index * rs + k * ms;
                    printf("dftw-genericbuf real part error, index %d real pos ~ val: %d ~ %f\n", index, pos, rio[pos]);
                    rio[pos] -= diff; 

               }
               diff = sumPos1[1] - outChecksum1[2*k+1];
               if(diff < -dftwDelta || diff > dftwDelta)
               {
                    printf("processor %d memory error in imaginary part of %d-th FFT bytwiddle, sum %.10f cks %.10f diff %.10f\n", myRank, k, sumPos1[1], outChecksum1[2*k+1], diff);
                    // fflush(stdout);
                    recalc = true;
                    R sum11 = 0;

                    for(int j=0; j<r; j++)
                    {
                         sum11 += (j+1) * iio[j*rs + k*ms];
                    }
                    R diff2 = sum11 - outChecksum2[2*k+1];

                    // int index =(int) (diff / diff2 - 0.5); // -1 for index
                    int index =(int) (diff2 / diff - 0.5); // -1 for index
                    // printf("sum, cks, diff2: %.10f %.10f %.10f\tindex: %d\n", sum11, outChecksum2[2*k+1], diff2, index);fflush(stdout);

                    int pos = index * rs + k * ms;
                    printf("dftw-genericbuf real part error, index %d real pos ~ val: %d ~ %f\n", index, pos, iio[pos]);
                    iio[pos] -= diff; 
               }
               if(recalc)
               {
                    break;
               }
               sumPos1 += 2;
          }

          dftwIPC[0] = ipcReal, dftwIPC[1] = ipcIm;
     }
}

static int applicable0(const S *ego,
		       INT r, INT irs, INT ors,
		       INT m, INT v,
		       INT mcount)
{
     return (1
	     && v == 1
	     && irs == ors
	     && mcount >= ego->batchsz
	     && mcount % ego->batchsz == 0
	     && r >= 64 
	     && m >= r
	  );
}

static int applicable(const S *ego,
		      INT r, INT irs, INT ors,
		      INT m, INT v,
		      INT mcount,
		      const planner *plnr)
{
     if (!applicable0(ego, r, irs, ors, m, v, mcount))
	  return 0;
     if (NO_UGLYP(plnr) && m * r < 65536)
	  return 0;

     return 1;
}

// checksum vector generation with DMR
static void dftw_rA_generation(int m, R * rA)
{
    R * rA0;
    R sumrA0 = 0, sumrA1 = 0;

     {
          int res = m % 3;
          register R c1 = 0.866025403784438818632906986749731004238128662;
          register R numeratorIm;
          if(res == 1) 
          {
               numeratorIm = - c1;
               rA[0] = 1, rA[1] = 0;
          }
          else 
          {
               numeratorIm = c1;
               rA[0] = 0.5, rA[1] = c1;  
          }
          rA0 = rA;
          rA0 += 2;
          int i;
           for(i=1; i<m; i++) // dmr
           {   
               R wi[2];
               real_cexp(i, m, wi);

               register R temp0, temp1, res0, res1, denom;
               register R w0, w1;
               register R dmr;
               w0 = wi[0], w1 = wi[1];
               // rA[m]
               temp0 = 1 + 0.5 * w0 - c1 * w1;
               dmr = 1 + 0.5 * w0 - c1 * w1;
               if(temp0!=dmr) 
                {
                    printf("rA error first part! temp0\n");
                    temp0 = 1 + 0.5 * w0 - c1 * w1;
                }
               
               temp1 = 0.5 * w1 + c1 * w0;
               dmr = 0.5 * w1 + c1 * w0;

               if(temp1!=dmr) 
                {
                    printf("rA error first part! temp1\n");
                    temp1 = 0.5 * w1 + c1 * w0;
                }
               // if(i==100)
               //      printf("temp: %f %f\n", temp0, temp1);

               res0 = 1.5 * temp0 - numeratorIm * temp1;
               dmr = 1.5 * temp0 - numeratorIm * temp1;
               if(res0!=dmr) 
                {
                    printf("rA error first part! res0\n");
                    res0 = 1.5 * temp0 - numeratorIm * temp1;
                }

                res1 = 1.5 * temp1 + numeratorIm * temp0;
                dmr = 1.5 * temp1 + numeratorIm * temp0;

               if(res1!=dmr) 
                {
                    printf("rA error first part! res1\n");
                    dmr = 1.5 * temp1 + numeratorIm * temp0;
                }

               denom = temp0 * temp0 + temp1 * temp1;
               dmr = temp0 * temp0 + temp1 * temp1;

               if(denom!=dmr) 
                {
                    printf("rA error first part! denom\n");
                   denom = temp0 * temp0 + temp1 * temp1;
                }

               w0 = res0 / denom;
               dmr = res0 / denom;
               if(w0!=dmr) 
                {
                    printf("rA error first part! w0\n");
                    w0 = res0 / denom;
                }

               w1 = res1 / denom;
               dmr = res1 / denom;
               if(w1!=dmr) 
                {
                    printf("rA error first part! w1:\n");                   
                    w1 = res1 / denom;
                }

               rA0[0] = w0;
               rA0[1] = w1;

               sumrA0 += w0;
               sumrA1 += w1;

               rA0 += 2;
           }
     }

     // checksum for rA
     rA0[0] = sumrA0;
     rA0[1] = sumrA1;

     return ;
}

static void dftw_output_verification(R * ro0, R * io0, int cn, R * ipc)
{
    R or00, or01, or10, or11, or20, or21;
    or00 = 0, or01 = 0, or10 = 0, or11 = 0, or20 = 0, or21 = 0; 

    int j;
    for(j=0; j<cn - 3; j+=3)
    {
        or00 += ro0[0], or01 += io0[0];
        or10 += ro0[2], or11 += io0[2];
        or20 += ro0[4], or21 += io0[4];
        ro0 += 6, io0 += 6;  
    }
    if((j+1) < cn )
    {
        or00 += ro0[0], or01 += io0[0];
        or10 += ro0[2], or11 += io0[2];
    }
    else
    {
        or00 += ro0[0], or01 += io0[0];
    }                      

    R r1 = 0.866025403784438818632906986749731004238128662;

    ipc[0] += - or00 + 0.5*or10 + r1*or11 + 0.5*or20 - r1*or21;
    ipc[1] += - or01 + 0.5*or11 - r1*or10 + 0.5*or21 + r1*or20; 

    return ;
}

R cks_diff;
static void dobatch(const P *ego, INT mb, INT me, R *buf, R *rio, R *iio)
{
     plan_dft *cld;
     INT ms = ego->ms;
     INT rs = ego->rs;

     int batchSize = ego->batchsz;
     int cn = ego->r;
     bool recalc = true;

     while(recalc)
     {

          bytwiddle(ego, mb, me, buf, rio, iio);

          cld = (plan_dft *) ego->cld;
          cld->apply(ego->cld, buf, buf + 1, buf, buf + 1);

          // inject fault
          // if(mb == 596 && !inject2)
          // {
          //   buf[3457] += 5398;
          //   inject2 = true;
          // }

          // printf("processor %d calculation done\n", myRank);
          // fflush(stdout);

          int offset;
          recalc = false;
          for(int i=0; i<batchSize; i++)
          {
               // printf("processor %d %d-th FFT verification done\n", myRank, i);
               // fflush(stdout);

               offset = 2 * BATCHDIST(cn) * i;
               dftw_output_verification(buf + offset, buf + offset + 1, cn, dftwIPC);
          }
          if(dftwIPC[0]<-dftwDelta2 || dftwIPC[0]>dftwDelta2 || dftwIPC[1]<-dftwDelta2 || dftwIPC[1]>dftwDelta2) 
          {
               printf("processor %d dftw-genericbuf error\n", myRank);
               // fflush(stdout);

               recalc = true;
          }
          // do not need correction here, since memory error can be corrected by bytwiddle while 
          // computational error would be corrected by re-calculation

     }
     cks_diff = fabs(dftwIPC[0]) > fabs(dftwIPC[1]) ? fabs(dftwIPC[0]) : fabs(dftwIPC[1]);

     // printf("processor %d %d~%d FFTs done\n", myRank, mb, me);
     // fflush(stdout);

     {
          int num = cn / pNum; // number of element that in a same transpose block and output buffer
          int stride = ego->me - ego->mb;
          int outputBufferStride = 2 * BATCHDIST(cn);

          // if(myRank == 0) printf("dftw-genericbuf final output stride: %d\n", stride);
          // if(myRank == 5)
          // {
          //      printf("ms mb ego->r ego->rs\n", ms, mb, ego->r, ego->rs);
          // }

          R * cksPos1 = outputChecksum1;
          R * cksPos2 = outputChecksum2;
          R * pos = buf;
          R * pos0;

          register R temp0, temp1;

          int cof, cof1, c;

          for (int i=0; i<pNum; i++)
          {
               cof = mb + 1;
               for(int j=0; j<num; j++)
               {
                    pos0 = pos;
                    for (int k=0; k<batchSize; k++) 
                    {
                         c = cof + k;
                         temp0 = pos0[0], temp1 = pos0[1];
                         cksPos1[0] += temp0;
                         cksPos1[1] += temp1;
                         cksPos2[0] += c * temp0;
                         cksPos2[1] += c * temp1;
                         pos0 += outputBufferStride;
                    }
                    cof += stride, pos += 2;
               }
               cksPos1 += 2, cksPos2 += 2;
          }
     }

     X(cpy2d_pair_co)(buf, buf + 1,
                rio + ms * mb, iio + ms * mb,
                me-mb, 2 * BATCHDIST(ego->r), ms,
                ego->r, 2, ego->rs);
}

static void apply(const plan *ego_, R *rio, R *iio)
{
     const P *ego = (const P *) ego_;
     R *buf = (R *) MALLOC(sizeof(R) * 2 * BATCHDIST(ego->r) * ego->batchsz,
			   BUFFERS);

     int radix = (ego->me - ego->mb);
     int cn = ego->r;
     int batchSize = ego->batchsz;
     // if dmr, regenerate checksums
     if(dmr)
     {
          // sleep for DMR
          clock_t t2 = clock();
          double eTime = (t2 - t1) / CLOCKS_PER_SEC;
          int uSleepTime = (int) (eTime * 3*1e5);
          usleep(uSleepTime);

          free(outChecksum1);
          free(outChecksum2);
          outChecksum1 = (R *) calloc(radix*2, sizeof(R));
          outChecksum2 = (R *) calloc(radix*2, sizeof(R));

          R * pos = rio;
          R * cksPos1;
          R * cksPos2;
          register R temp0, temp1;
          int cof;
          for(int i=0; i<cn; i++)
          {
               R * cksPos1 = outChecksum1;
               R * cksPos2 = outChecksum2;
               cof = i+1;
               for(int j=0; j<radix; j++)
               {
                    temp0 = pos[0], temp1 = pos[1];
                    cksPos1[0] += temp0;
                    cksPos1[1] += temp1;
                    cksPos2[0] += cof * temp0;
                    cksPos2[1] += cof * temp1;
                    pos += 2, cksPos1 += 2, cksPos2 += 2;
               }
          }
     }

     if(cn * fft2Radix != myRadix)
     {
          free(rA);
          rA = (R *) malloc((cn+2)*2*sizeof(R));
          dftw_rA_generation(cn, rA);
     }


     // outputChecksum1 = (R *) calloc(2*radix, sizeof(R));
     // outputChecksum2 = (R *) calloc(2*radix, sizeof(R));     

     outputChecksum1 = (R *) calloc(2*pNum, sizeof(R));
     outputChecksum2 = (R *) calloc(2*pNum, sizeof(R));     

     dftwIPC = (R *) malloc(2*sizeof(R));
     interChecksum1 = (R *) malloc(2*batchSize*sizeof(R));

     // inject error
     // iio[22891175] += 22363893;

     // if(myRank == 74) 
     // {
     //      rio[2477] += 180;
     //      iio[36325] += 48;
     // }
     


     INT m;
     int k = ego->r;
     // R threshold1 = 3 * sqrt(k) * k * sqrt(2.0*pNum*myRadix*1.0/3*log2(k))*sqrt(0.21)*2.220446049E-16; //uniform distribution
     R threshold1 = 3 * sqrt(k) * k * sqrt(2.0*pNum*myRadix*1.0*log2(k))*sqrt(0.21)*2.220446049E-16; //normal distribution
     int exceed_count = 0;
     R max_diff = 0;

     for (m = ego->mb; m < ego->me; m += ego->batchsz){
	     dobatch(ego, m, m + ego->batchsz, buf, rio, iio);
	     if(cks_diff > threshold1) exceed_count ++;
	     if(cks_diff > max_diff) max_diff = cks_diff;
     }
     printf("FFT3 threshold=%.4e, max_diff=%.4e\n", threshold1, max_diff, myRadix);
     printf("FFT3 exceed_count=%d, total=%d, percent=%.4f\n", exceed_count, (ego->me - ego->mb)/ego->batchsz, exceed_count * 1.0/((ego->me - ego->mb)/ego->batchsz));

     A(m == ego->me);



     free(dftwIPC);
     free(rA);
     free(interChecksum1);

     // printf("processor %d dftw-genericbuf free done\n", myRank);
     // fflush(stdout);
     // if(myRank == 10)
     // {
          // R * pos = rio + 32768*2;
          // for(int i=0; i<20; i++)
          // {
          //      printf("%d: %f %f\n", i, pos[0], pos[1]);
          //      pos += 2;
          // }
          // printf("\n\n\n");
     //      R * pos = rio;
     //      for(int i=0; i<pNum; i++)
     //      {
     //          R sum10 = 0, sum11 = 0;
     //          for(int j=1; j <= myRadix/pNum; j++)
     //          {
     //              sum10 += j*pos[0];
     //              sum11 += j*pos[1];
     //              pos += 2;
     //          }
     //          printf("sum: %.10f %.10f\tcks: %.10f %.10f\n", sum10, sum11, outputChecksum2[2*i], outputChecksum2[2*i+1]);
     //      }
     // }

     X(ifree)(buf);
}

static void awake(plan *ego_, enum wakefulness wakefulness)
{
     P *ego = (P *) ego_;
     X(plan_awake)(ego->cld, wakefulness);

     switch (wakefulness) {
	 case SLEEPY:
	      X(triggen_destroy)(ego->t); ego->t = 0;
	      break;
	 default:
	      ego->t = X(mktriggen)(AWAKE_SQRTN_TABLE, ego->r * ego->m);
	      break;
     }
}

static void destroy(plan *ego_)
{
     P *ego = (P *) ego_;
     X(plan_destroy_internal)(ego->cld);
}

static void print(const plan *ego_, printer *p)
{
     const P *ego = (const P *) ego_;
     p->print(p, "(dftw-genericbuf/%D-%D-%D%(%p%))",
	      ego->batchsz, ego->r, ego->m, ego->cld);
}

static plan *mkcldw(const ct_solver *ego_,
		    INT r, INT irs, INT ors,
		    INT m, INT ms,
		    INT v, INT ivs, INT ovs,
		    INT mstart, INT mcount,
		    R *rio, R *iio,
		    planner *plnr)
{
     const S *ego = (const S *)ego_;
     P *pln;
     plan *cld = 0;
     R *buf;

     static const plan_adt padt = {
	  0, awake, print, destroy
     };
     
     UNUSED(ivs); UNUSED(ovs); UNUSED(rio); UNUSED(iio);

     A(mstart >= 0 && mstart + mcount <= m);
     if (!applicable(ego, r, irs, ors, m, v, mcount, plnr))
          return (plan *)0;

     buf = (R *) MALLOC(sizeof(R) * 2 * BATCHDIST(r) * ego->batchsz, BUFFERS);
     cld = X(mkplan_d)(plnr,
			X(mkproblem_dft_d)(
			     X(mktensor_1d)(r, 2, 2),
			     X(mktensor_1d)(ego->batchsz,
					    2 * BATCHDIST(r),
					    2 * BATCHDIST(r)),
			     buf, buf + 1, buf, buf + 1
			     )
			);
     X(ifree)(buf);
     if (!cld) goto nada;

     pln = MKPLAN_DFTW(P, &padt, apply);
     pln->slv = ego;
     pln->cld = cld;
     pln->r = r;
     pln->m = m;
     pln->ms = ms;
     pln->rs = irs;
     pln->batchsz = ego->batchsz;
     pln->mb = mstart;
     pln->me = mstart + mcount;

     {
	  double n0 = (r - 1) * (mcount - 1);
	  pln->super.super.ops = cld->ops;
	  pln->super.super.ops.mul += 8 * n0;
	  pln->super.super.ops.add += 4 * n0;
	  pln->super.super.ops.other += 8 * n0;
     }
     return &(pln->super.super);

 nada:
     X(plan_destroy_internal)(cld);
     return (plan *) 0;
}

static void regsolver(planner *plnr, INT r, INT batchsz)
{
     S *slv = (S *)X(mksolver_ct)(sizeof(S), r, DECDIT, mkcldw, 0);
     slv->batchsz = batchsz;
     REGISTER_SOLVER(plnr, &(slv->super.super));

     if (X(mksolver_ct_hook)) {
	  slv = (S *)X(mksolver_ct_hook)(sizeof(S), r, DECDIT, mkcldw, 0);
	  slv->batchsz = batchsz;
	  REGISTER_SOLVER(plnr, &(slv->super.super));
     }

}

void X(ct_genericbuf_register)(planner *p)
{
     static const INT radices[] = { -1, -2, -4, -8, -16, -32, -64 };
     static const INT batchsizes[] = { 4, 8, 16, 32, 64 };
     unsigned i, j;

     for (i = 0; i < sizeof(radices) / sizeof(radices[0]); ++i)
	  for (j = 0; j < sizeof(batchsizes) / sizeof(batchsizes[0]); ++j)
	       regsolver(p, radices[i], batchsizes[j]);
}
