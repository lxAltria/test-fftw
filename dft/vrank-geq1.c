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



/* Plans for handling vector transform loops.  These are *just* the
   loops, and rely on child plans for the actual DFTs.
 
   They form a wrapper around solvers that don't have apply functions
   for non-null vectors.
 
   vrank-geq1 plans also recursively handle the case of multi-dimensional
   vectors, obviating the need for most solvers to deal with this.  We
   can also play games here, such as reordering the vector loops.
 
   Each vrank-geq1 plan reduces the vector rank by 1, picking out a
   dimension determined by the vecloop_dim field of the solver. */

#include "dft.h"

#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <time.h>

extern bool firstFFT;
extern bool secondFFT;
extern int myRadix;
extern int myRank;
extern int pNum;
extern int fft2Radix;
extern bool dmr;

extern R * checksum1;
extern R * checksum2;
extern R * rA;
extern R * rAcoeff;
extern R * outChecksum1;
extern R * outChecksum2;
extern R * fft2checksum1;
extern R * fft2checksum2;

extern clock_t t1;

bool inject11 = false;
bool inject12 = false;

typedef struct {
     solver super;
     int vecloop_dim;
     const int *buddies;
     size_t nbuddies;
} S;

typedef struct {
     plan_dft super;

     plan *cld;
     INT vl;
     INT ivs, ovs;
     const S *solver;
} P;

// static void apply(const plan *ego_, R *ri, R *ii, R *ro, R *io)
// {
//      const P *ego = (const P *) ego_;
//      INT i, vl = ego->vl;
//      INT ivs = ego->ivs, ovs = ego->ovs;
//      dftapply cldapply = ((plan_dft *) ego->cld)->apply;

//      for (i = 0; i < vl; ++i) {
//           cldapply(ego->cld,
//                    ri + i * ivs, ii + i * ivs, ro + i * ovs, io + i * ovs);
//      }
// }

R cks_diff;
static bool output_verification_continuous(R * ro0, R * io0, int cn, R * cks1, R * diff, R delta)
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

    diff[0] = cks1[0] - or00 + 0.5*or10 + r1*or11 + 0.5*or20 - r1*or21;
    diff[1] = cks1[1] - or01 + 0.5*or11 - r1*or10 + 0.5*or21 + r1*or20; 
    cks_diff = fabs(diff[0]) > fabs(diff[1]) ? fabs(diff[0]) : fabs(diff[1]);
    return (diff[0]>delta || diff[0]<-delta || diff[1]>delta || diff[1]<-delta);

}

static void error_correction_discontinuous(R * ri0, R * ii0, int m, int r, R * coeff, R * rA, R * cks2, R * diff, R * lastResult, R delta)
{
    //recalc 2 times, correct memory here
    // printf("memory error! ivs: %d\tovs: %d\n", ivs, ovs);
    R * in1;
    R * in2;
    R * cof;
    in1 = ri0, in2 = ii0;
    cof = coeff;

    // current memory checksums
    R cksMR = 0, cksMI = 0;
    int step = 2 * r;

    for(int j=0; j<m; j++)
    {
        cksMR += cof[0] * in1[0];
        cksMI += cof[1] * in2[0];
        in1 += step, in2 += step;
        cof += 2;
    }

    R diffReal = diff[0];
    R diffIm = diff[1];

    R diffReal2 = cks2[0] - cksMR;
    R diffIm2 = cks2[1] - cksMI;

    printf("diff2 : %f %f\n", diffReal2, diffIm2);
    
    if(fabs(diffReal2) < 1e-9 && fabs(diffIm2) < 1e-9)  
    {
      printf("computational error or x[1] corruption!\n");
      if(diffReal == lastResult[0] && diffIm == lastResult[1])
      {
        printf("x[1] corruption, %f\n", ii0[0]);
        ii0[0] += diffIm;
      }
      return ;
    }
    printf("memory error!\n");
    if(fabs(diffReal2) < 1e-9)
    {
        printf("imaginary part error!");
        // int index =(int) (-diffReal / diffIm2 - 0.5); // -1 for index
        int index =(int) (-diffIm2 / diffReal - 0.5); // -1 for index
        int pos = index * r * 2;
        printf("memory error ocurrs imaginary part, pos -- val: %d -- %f\n", pos, ii0[pos]);
        ii0[pos] -= diffReal / rA[2*index+1]; 
        printf("index %d diff: %f\n", index, diffReal / rA[2*index+1]);
    }
    else if(fabs(diffIm2) < 1e-9)
    {
        printf("real part error!");
        // int index =(int) (diffReal / diffReal2 - 0.5); // -1 for index
        int index =(int) (diffReal2 / diffReal - 0.5); // -1 for index
        int pos = index * r * 2; // real part
        printf("memory error ocurrs real part, pos -- val: %d -- %f\n", pos, ri0[pos]);
        ri0[pos] -= -diffReal / rA[2*index]; 
        printf("index %d diff: %f\n", index, -diffReal / rA[2*index]);
    }
    else
    {
        printf("multiple memory error, restart is needed.\n");
        fflush(stdout);
        exit(1);
    }
    return ;
}

static void error_correction_continuous(R * ri0, R * ii0, int m, R * coeff, R * rA, R * cks2, R * diff, R * lastResult, R delta)
{
    //recalc 2 times, correct memory here
    // printf("memory error! ivs: %d\tovs: %d\n", ivs, ovs);
    R * in1;
    R * in2;
    R * cof;
    in1 = ri0, in2 = ii0;
    cof = coeff;

    // current memory checksums
    R cksMR = 0, cksMI = 0;

    for(int j=0; j<m; j++)
    {
        cksMR += cof[0] * in1[0];
        cksMI += cof[1] * in2[0];
        in1 += 2, in2 += 2;
        cof += 2;
    }

    R diffReal = diff[0];
    R diffIm = diff[1];

    R diffReal2 = cks2[0] - cksMR;
    R diffIm2 = cks2[1] - cksMI;

    printf("diff2 : %f %f\n", diffReal2, diffIm2);
    
    if(fabs(diffReal2) < delta && fabs(diffIm2) < delta)  
    {
      printf("computational error or x[1] corruption!\n");
      if(diffReal == lastResult[0] && diffIm == lastResult[1])
      {
        printf("x[1] corruption, %f\n", ii0[0]);
        ii0[0] += diffIm;
      }
      return ;
    }
    printf("memory error!\n");
    if(fabs(diffReal2) < delta)
    {
        printf("imaginary part error!");
        // int index =(int) (-diffReal / diffIm2 - 0.5); // -1 for index
        int index =(int) (-diffIm2 / diffReal - 0.5); // -1 for index
        int pos = index * 2;
        printf("memory error ocurrs imaginary part, pos -- val: %d -- %f\n", pos, ii0[pos]);
        ii0[pos] -= diffReal / rA[2*index+1]; 
        printf("index %d diff: %f\n", index, diffReal / rA[2*index+1]);
    }
    else if(fabs(diffIm2) < delta)
    {
        printf("real part error!");
        // int index =(int) (diffReal / diffReal2 - 0.5); // -1 for index
        int index =(int) (diffReal2 / diffReal - 0.5); // -1 for index
        int pos = index * 2; // real part
        printf("memory error ocurrs real part, pos -- val: %d -- %f\n", pos, ri0[pos]);
        ri0[pos] -= -diffReal / rA[2*index]; 
        printf("index %d diff: %f\n", index, -diffReal / rA[2*index]);
    }
    else
    {
        printf("multiple memory error, restart is needed.\n");
        fflush(stdout);
        exit(1);
    }
    return ;
}

static void apply(const plan *ego_, R *ri, R *ii, R *ro, R *io)
{
     const P *ego = (const P *) ego_;
     INT i, vl = ego->vl;
     INT ivs = ego->ivs, ovs = ego->ovs;
     dftapply cldapply = ((plan_dft *) ego->cld)->apply;

     if(firstFFT)
     {
        if(vl * pNum == myRadix)
        {
            // if(myRank == 0) printf("firstFFT starts, vl and m: %d %d\n", vl, pNum);
            // fflush(stdout);
            int batchSize = 4;

            int m = pNum;
            int radix = vl;
            R * inputBuffer = (R *) malloc(m*2*batchSize*sizeof(R));
            R * outputBuffer = (R *) malloc(m*2*sizeof(R));
            // outChecksum with initialization
            outChecksum1 = (R *) calloc(m*2, sizeof(R));
            outChecksum2 = (R *) calloc(m*2, sizeof(R));
            // outChecksum1 = (R *) malloc(m*2*sizeof(R));
            // outChecksum2 = (R *) malloc(m*2*sizeof(R));
            // for(int i=0; i<m*2; i++)
            // {
            //     outChecksum1[i] = 0;
            //     outChecksum2[i] = 0;
            // }
            // printf("outChecksum: %.15f %.15f\n", outChecksum1[2*255], outChecksum1[2*255+1]);
           
            int stepSize = vl * 2;
            R * pos;
            R * ib;
            R * cksPos1 = checksum1;
            R * cksPos2 = checksum2;

            R delta1 = 1e-2;
	    // R threshold1 = 3 * sqrt(pNum) * pNum * sqrt(2*pNum*1.0/3*log2(pNum))*sqrt(0.21)*2.220446049E-16; //uniform distribution
	    R threshold1 = 3 * sqrt(pNum) * pNum * sqrt(2*pNum*1.0*log2(pNum))*sqrt(0.21)*2.220446049E-16; //normal distribution
	    int exceed_count = 0;
	    R max_diff = 0;

            int bufferOffset = 2*m;
            // inject fault
            // if(myRank == 0) ri[346453] += 500;
            // ri[1236534] += 12556;

            for (i = 0; i < vl; i += 4) 
            {
                
                R diff[2];
                R lastResult[2] = {-10000, -10000};

                // for(int bs = 0; bs<batchSize; bs++)
                // {
                //     pos = ri + (i+bs)*ivs;
                //     ib = inputBuffer + bs * 2*m;
                //     for(int j=0; j<m; j++)
                //     {
                //         ib[0] = pos[0];
                //         ib[1] = pos[1];
                //         ib += 2, pos += stepSize;
                //     } 
                // }
                {
                    pos = ri + i*ivs;
                    ib = inputBuffer;
                    R * ib2;
                    for(int j=0; j<m; j++)
                    {
                        ib2 = ib;
                        for(int bs = 0; bs<batchSize; bs++)
                        {
                            ib2[0] = pos[0];
                            ib2[1] = pos[1];
                            ib2 += bufferOffset, pos += 2;
                        }
                        ib += 2, pos += stepSize - 2*batchSize;
                    } 
                }

                R * input = inputBuffer;
                for(int bs = 0; bs<batchSize; bs++)
                {

                    bool recalc = true;
                    while(recalc)
                    {

                        cldapply(ego->cld, input, input+1, outputBuffer, outputBuffer+1);

                        // inject fault
                        // if(i==340 && !inject11)
                        // {
                        //     outputBuffer[72] += 734;
                        //     inject11 = true;
                        // }

                        recalc = output_verification_continuous(outputBuffer, outputBuffer+1, m, cksPos1, diff, delta1);

                        if(recalc) 
                        {
                            printf("processor %d diff in %d-th firstFFT: %f %f, pos offset: %lld\n", myRank, i+bs, diff[0], diff[1], (long long)((i+bs)*ivs));
                            error_correction_discontinuous(ri + (i+bs) * ivs, ii + (i+bs) * ivs, m, radix, rAcoeff, rA, cksPos2, diff, lastResult, delta1);
                            R * ib = input;
                            R * pos = ri + (i+bs)*ivs;
                            for(int j=0; j<m; j++)
                            {
                                ib[0] = pos[0];
                                ib[1] = pos[1];
                                ib += 2, pos += stepSize;
                            }

                            lastResult[0] = diff[0];
                            lastResult[1] = diff[1];
                            continue;
                        }
                    }
		    if(cks_diff > threshold1) exceed_count ++;
		    if(cks_diff > max_diff) max_diff = cks_diff;
                    // computation is correct
                    pos = ro + (i+bs)*ovs;
                    ib = outputBuffer;
                    register R temp0, temp1;
                    R * oCksPos1 = outChecksum1;
                    R * oCksPos2 = outChecksum2;

                    int index = i+bs;
                    for(int j=0; j<m; j++)
                    {
                        temp0 = ib[0], temp1 = ib[1];
                        pos[0] = temp0;
                        pos[1] = temp1;
                        oCksPos1[0] += temp0, oCksPos1[1] += temp1;
                        oCksPos2[0] += index*temp0, oCksPos2[1] += index*temp1;
                        pos += stepSize, ib += 2;
                        oCksPos1 += 2, oCksPos2 += 2;
                    }

                    cksPos1 += 2;
                    cksPos2 += 2;

                    input += bufferOffset;
                }

            }
	    printf("FFT1 threshold=%.4e, max_diff=%.4e\n", threshold1, max_diff);
	    printf("FFT1 exceed_count=%d, total=%d, percent=%.4f\n", exceed_count, vl, exceed_count * 1.0/vl);
            free(rA);
            free(rAcoeff);
            free(checksum1);
            free(checksum2);
            free(inputBuffer);
            free(outputBuffer);
        }
        else
        {
             for (i = 0; i < vl; ++i) {
                      cldapply(ego->cld,
                               ri + i * ivs, ii + i * ivs, ro + i * ovs, io + i * ovs);
                 }             
        }
     }
     else if(secondFFT)
     {
        // if(myRank == 0) printf("fft2 start, vl %d fft2Radix %d\n", vl, fft2Radix);
        
        if(vl == fft2Radix)
        {
            int m = myRadix / vl;
            // if(myRank == 0) printf("fft2 start, radix %d m %d myRadix %d\n", vl, m, myRadix);
            // fflush(stdout);


            R * inputBuffer = (R *) malloc(m*2*sizeof(R));

            // R * cksPos1 = fft2checksum1;
            // R * cksPos2 = fft2checksum2;

            outChecksum1 = (R *) calloc(m*2, sizeof(R));
            outChecksum2 = (R *) calloc(m*2, sizeof(R));
            R delta2 = 1e-1;
	    //R threshold1 = 3 * sqrt(m) * m * sqrt(2*pNum*m*1.0/3*log2(m))*sqrt(0.21)*2.220446049E-16; //uniform distribution
	    R threshold1 = 3 * sqrt(m) * m * sqrt(2*pNum*m*1.0*log2(m))*sqrt(0.21)*2.220446049E-16; //normal distribution
            int exceed_count = 0;
            R max_diff = 0;
            // inject fault
            // if(myRank == 0) ri[2724886] += 500;     
            // ri[324688] += 3463;       

            int ratio = vl / m;

         for (i = 0; i < vl; ++i) {

                R diff[2];
                R lastResult[2] = {-10000, -10000};

                bool recalc = true;

                memcpy(inputBuffer, ri + i*ivs, 2*m*sizeof(R));
                    
                while(recalc)
                {            
                    cldapply(ego->cld,
                           ri + i * ivs, ii + i * ivs, ro + i * ovs, io + i * ovs);

                    // inject fault
                    // if(i==164 && !inject12)
                    // {
                    //     (ro + i * ovs)[623] += 5684;
                    //     inject12 = true;
                    // }

                    recalc = output_verification_continuous(ro + i * ovs, io + i * ovs, m, fft2checksum1 + 2*((i%ratio)*m+i/ratio), diff, delta2);
                    if(recalc) 
                    {
                        printf("processor %d diff in %d-th secondFFT: %f %f, pos offset: %lld\n", myRank, i, diff[0], diff[1], (long long)i*ivs);
                        // fflush(stdout);
                        // exit(1);
                        // correct both input buffer and input
                        error_correction_continuous(inputBuffer, inputBuffer+1, m, rAcoeff, rA, fft2checksum2 + 2*((i%ratio)*m+i/ratio), diff, lastResult, delta2);
                        memcpy(ri + i*ivs, inputBuffer, 2*m*sizeof(R));
                        lastResult[0] = diff[0];
                        lastResult[1] = diff[1];
                        continue;
                    }

                } // end while
		if(cks_diff > threshold1) exceed_count ++;
                if(cks_diff > max_diff) max_diff = cks_diff;
                R * pos = ro + i * ovs;
                R * oCksPos1 = outChecksum1;
                R * oCksPos2 = outChecksum2;
                register R temp0, temp1;

                int index = i+1;
                for(int j=0; j<m; j++)
                {
                    temp0 = pos[0], temp1 = pos[1];
                    oCksPos1[0] += temp0;
                    oCksPos1[1] += temp1;
                    oCksPos2[0] += index * temp0;
                    oCksPos2[1] += index * temp1;
                    pos += 2;
                    oCksPos1 += 2, oCksPos2 += 2;
                }
             }  
	     printf("FFT2 threshold=%.4e, max_diff=%.4e\n", threshold1, max_diff);
             printf("FFT2 exceed_count=%d, total=%d, percent=%.4f\n", exceed_count, vl, exceed_count * 1.0/vl);
             free(fft2checksum1);
             free(fft2checksum2);
             free(rAcoeff);
             free(inputBuffer);

             // if(myRank == 0) printf("\n\n\n");

             if(dmr)    // check memory
             {
                // if(myRank == 0) printf("check memory\n");

                //inject fault
                // ro[623471] += 2367;

                R * pos = ro;
                R diff;

                R * sumRo = (R * ) calloc(2*m, sizeof(R));
                R * posSum;
                for(int i=0; i<vl; i++)
                {
                    posSum = sumRo;
                    for(int j=0; j<m; j++)
                    {
                        posSum[0] += pos[0], posSum[1] += pos[1];
                        pos += 2, posSum += 2;
                    }
                }
                // if(myRank == 120) printf("outChecksum1: %f %f\n", outChecksum1[72], outChecksum1[73]);
                R * oCksPos1 = outChecksum1;
                posSum = sumRo;
                for(int i=0; i<m; i++)
                {
                    diff = posSum[0] - oCksPos1[0];
                    if(diff < -delta2 || diff > delta2)
                    {
                        printf("processor %d error before dmr in real part, %d-th FFT sum %.10f cks %.10f diff %.10f\n", myRank, i, sumRo[0], oCksPos1[0], diff);
                        // corretion
                        R sum10 = 0;
                        int stepSize = 2*m;
                        R * pos0 = ro + 2*i;
                        for(int j=1; j<=vl; j++)
                        {
                            sum10 += j*pos0[0];
                            pos0 += stepSize;
                        }
                        R diff2 = sum10 - outChecksum2[2*i];

                        int index =(int) (diff2 / diff - 0.5); // -1 for index

                        int pos = index * stepSize + 2*i;
                        printf("before dmr real part error, index %d real pos ~ val: %d ~ %f\n", index, pos, ro[pos]);
                        ro[pos] -= diff;                         
                        // exit(0);
                    }
                    diff = posSum[1] - oCksPos1[1];
                    if(diff < -delta2 || diff > delta2)
                    {
                        printf("processor %d error before dmr in imaginary part, %d-th FFT sum %.10f cks %.10f diff %.10f\n", myRank, i, sumRo[1], oCksPos1[1], diff);
                        // correction
                        R sum11 = 0;
                        int stepSize = 2*m;
                        R * pos0 = io + 2*i;
                        for(int j=1; j<=vl; j++)
                        {
                            sum11 += j*pos0[0];
                            pos0 += stepSize;
                        }
                        R diff2 = sum11 - outChecksum2[2*i+1];

                        int index =(int) (diff2 / diff - 0.5); // -1 for index

                        int pos = index * stepSize + 2*i;
                        printf("before dmr imaginary part error, index %d imaginary pos ~ val: %d ~ %f\n", index, pos, io[pos]);
                        io[pos] -= diff; 
                        // exit(0);
                    }
                    oCksPos1 += 2, posSum += 2;
                }

                t1 = clock();
             }

        }
        else
        {
         for (i = 0; i < vl; ++i) {
                  cldapply(ego->cld,
                           ri + i * ivs, ii + i * ivs, ro + i * ovs, io + i * ovs);
             } 
        }
     }
     else
     {
         for (i = 0; i < vl; ++i) {
              cldapply(ego->cld,
                       ri + i * ivs, ii + i * ivs, ro + i * ovs, io + i * ovs);
         }        
     }
}

static void awake(plan *ego_, enum wakefulness wakefulness)
{
     P *ego = (P *) ego_;
     X(plan_awake)(ego->cld, wakefulness);
}

static void destroy(plan *ego_)
{
     P *ego = (P *) ego_;
     X(plan_destroy_internal)(ego->cld);
}

static void print(const plan *ego_, printer *p)
{
     const P *ego = (const P *) ego_;
     const S *s = ego->solver;
     p->print(p, "(dft-vrank>=1-x%D/%d%(%p%))",
          ego->vl, s->vecloop_dim, ego->cld);
}

static int pickdim(const S *ego, const tensor *vecsz, int oop, int *dp)
{
     return X(pickdim)(ego->vecloop_dim, ego->buddies, ego->nbuddies,
               vecsz, oop, dp);
}

static int applicable0(const solver *ego_, const problem *p_, int *dp)
{
     const S *ego = (const S *) ego_;
     const problem_dft *p = (const problem_dft *) p_;

     return (1
         && FINITE_RNK(p->vecsz->rnk)
         && p->vecsz->rnk > 0

         /* do not bother looping over rank-0 problems,
        since they are handled via rdft */
         && p->sz->rnk > 0

         && pickdim(ego, p->vecsz, p->ri != p->ro, dp)
      );
}

static int applicable(const solver *ego_, const problem *p_, 
              const planner *plnr, int *dp)
{
     const S *ego = (const S *)ego_;
     const problem_dft *p;

     if (!applicable0(ego_, p_, dp)) return 0;

     /* fftw2 behavior */
     if (NO_VRANK_SPLITSP(plnr) && (ego->vecloop_dim != ego->buddies[0]))
      return 0;

     p = (const problem_dft *) p_;

     if (NO_UGLYP(plnr)) {
      /* Heuristic: if the transform is multi-dimensional, and the
         vector stride is less than the transform size, then we
         probably want to use a rank>=2 plan first in order to combine
         this vector with the transform-dimension vectors. */
      {
           iodim *d = p->vecsz->dims + *dp;
           if (1
           && p->sz->rnk > 1 
           && X(imin)(X(iabs)(d->is), X(iabs)(d->os)) 
           < X(tensor_max_index)(p->sz)
            )
            return 0;
      }

      if (NO_NONTHREADEDP(plnr)) return 0; /* prefer threaded version */
     }

     return 1;
}

static plan *mkplan(const solver *ego_, const problem *p_, planner *plnr)
{
     const S *ego = (const S *) ego_;
     const problem_dft *p;
     P *pln;
     plan *cld;
     int vdim;
     iodim *d;

     static const plan_adt padt = {
      X(dft_solve), awake, print, destroy
     };

     if (!applicable(ego_, p_, plnr, &vdim))
          return (plan *) 0;
     p = (const problem_dft *) p_;

     d = p->vecsz->dims + vdim;

     A(d->n > 1);
     // cld = X(mkplan_d)(plnr,
           //     X(mkproblem_dft_d)(
              //   X(tensor_copy)(p->sz),
              //   X(tensor_copy_except)(p->vecsz, vdim),
              //   TAINT(p->ri, d->is), TAINT(p->ii, d->is),
              //   TAINT(p->ro, d->os), TAINT(p->io, d->os)));
     if(firstFFT && p->sz->dims[0].n == pNum && p->sz->dims[0].n * p->vecsz->dims[0].n == myRadix)
     {
        // if(myRank == 0) printf("d->n, sz, vecsz, myRadix: %d %d %d %d\n", d->n, p->sz->dims[0].n, p->vecsz->dims[0].n, myRadix);
        R * inputBuffer = malloc(sizeof(R)*2*p->sz->dims[0].n);
        R * outputBuffer = malloc(sizeof(R)*2*p->sz->dims[0].n);

        cld = X(mkplan_d)(plnr,
            X(mkproblem_dft_d)(
                 X(mktensor_1d)(p->sz->dims[0].n, 2, 2),
                 X(mktensor_1d)(1, 2*p->sz->dims[0].n, 2*p->sz->dims[0].n),
                 inputBuffer, inputBuffer + 1, outputBuffer, outputBuffer + 1
                 )
            );
        free(inputBuffer);
        free(outputBuffer);
     }
     else
     {
             cld = X(mkplan_d)(plnr,
               X(mkproblem_dft_d)(
                X(tensor_copy)(p->sz),
                X(tensor_copy_except)(p->vecsz, vdim),
                TAINT(p->ri, d->is), TAINT(p->ii, d->is),
                TAINT(p->ro, d->os), TAINT(p->io, d->os)));
     }

     if (!cld) return (plan *) 0;

     pln = MKPLAN_DFT(P, &padt, apply);

     pln->cld = cld;
     pln->vl = d->n;
     pln->ivs = d->is;
     pln->ovs = d->os;

     pln->solver = ego;
     X(ops_zero)(&pln->super.super.ops);
     pln->super.super.ops.other = 3.14159; /* magic to prefer codelet loops */
     X(ops_madd2)(pln->vl, &cld->ops, &pln->super.super.ops);

     if (p->sz->rnk != 1 || (p->sz->dims[0].n > 64))
      pln->super.super.pcost = pln->vl * cld->pcost;

     return &(pln->super.super);
}

static solver *mksolver(int vecloop_dim, const int *buddies, size_t nbuddies)
{
     static const solver_adt sadt = { PROBLEM_DFT, mkplan, 0 };
     S *slv = MKSOLVER(S, &sadt);
     slv->vecloop_dim = vecloop_dim;
     slv->buddies = buddies;
     slv->nbuddies = nbuddies;
     return &(slv->super);
}

void X(dft_vrank_geq1_register)(planner *p)
{
     /* FIXME: Should we try other vecloop_dim values? */
     static const int buddies[] = { 1, -1 };
     size_t i;
     
     for (i = 0; i < NELEM(buddies); ++i)
          REGISTER_SOLVER(p, mksolver(buddies[i], buddies, NELEM(buddies)));
}
