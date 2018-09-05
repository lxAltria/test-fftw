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

/* Distributed transposes using a sequence of carefully scheduled
   pairwise exchanges.  This has the advantage that it can be done
   in-place, or out-of-place while preserving the input, using buffer
   space proportional to the local size divided by the number of
   processes (i.e. to the total array size divided by the number of
   processes squared). */

#include "mpi-transpose.h"
#include "mpi-dft.h"
#include "dft.h"
#include <string.h>

#include <stdio.h>
#include <stdbool.h>

#include <unistd.h>

extern bool firstTranspose;
extern bool secondTranspose;
extern int myRank;
extern int pNum;
extern int fft2Radix;
extern int myRadix;

extern const plan * dft_rank1_plan;

extern R * checksum1;
extern R * checksum2;

extern R * rA;
extern R * rAcoeff;
extern R * outChecksum1;
extern R * outChecksum2;
extern R * fft2checksum1;
extern R * fft2checksum2;
extern R * outputChecksum1;
extern R * outputChecksum2;

typedef struct {
     solver super;
     int preserve_input; /* preserve input even if DESTROY_INPUT was passed */
} S;

typedef struct {
     plan_mpi_transpose super;

     plan *cld1, *cld2, *cld2rest, *cld3;
     INT rest_Ioff, rest_Ooff;
     
     int n_pes, my_pe, *sched;
     INT *send_block_sizes, *send_block_offsets;
     INT *recv_block_sizes, *recv_block_offsets;
     MPI_Comm comm;
     int preserve_input;
} P;

typedef struct {
     plan_mpi_dft super;

     triggen *t;
     plan *cldt, *cld_ddft, *cld_dft;
     INT roff, ioff;
     int preserve_input;
     INT vn, xmin, xmax, xs, m, r;
} dftr1P;

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

// checksum vector generation with DMR
static void checksumVectorRAGeneration(int m, R * rA)
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

// checksum rearrangement, coeff[2*i] = 1/i * rA[i]
static void coefficientGeneration(int m, R * rA, R * coeff)
{
    R * cof = coeff;
    R * posrA = rA;
    register R c;
    int i;
    for(i=1; i<=m; i++)
    {
        cof[0] = posrA[0] * i;
        cof[1] = posrA[1] * i;
        cof += 2, posrA += 2;
    }
    return ;
}

static void naive_checksum_calculation(R * data, int size, R * result)
{
	R sum00 = 0, sum01 = 0, sum10 = 0, sum11 = 0;
	R * pos = data;

	int i;
	int num = size/2;

	for(i=1; i<=num; i++)
	{
		sum00 += pos[0], sum10 += i * pos[0];
		sum01 += pos[1], sum11 += i * pos[1];
		pos += 2;
	}

	result[0] = sum00;
	result[1] = sum01;
	result[2] = sum10;
	result[3] = sum11;

	return;
}

static void transpose1_send_buffer_init(R * send, int size, R * cks1, R * cks2)
{
	R * pos = send + size;
	pos[0] = cks1[0];
	pos[1] = cks1[1];
	pos[2] = cks2[0];
	pos[3] = cks2[1];
	return ;
}

static void naive_checksum_verification(R * data, R delta, int size, R * result)
{
	R sum00 = 0, sum01 = 0;
	R * pos = data;
	int i;
	int num = size/2;
	for(i=0; i<num; i++)
	{
		sum00 += pos[0];// sum10 += cof[0] * pos[0];
		sum01 += pos[1];// sum11 += cof[0] * pos[1];
		pos += 2;
	}
	R diff = result[0] - sum00;
	if(diff<-delta || diff > delta)
	{
		printf("processor %d real part error! %.15f %.15f diff %.10f\n", myRank, result[0], sum00, diff);
    	// fflush(stdout);
		//corretion
    	pos = data;
    	R sum10 = 0;

    	for(i=1; i<=num; i++)
    	{
    		sum10 += i * pos[0];
    		pos += 2;
    	}
    	R diff2 = result[2] - sum10;
    	// int index = (int)(diff / diff2 - 0.5);
    	int index = (int)(diff2 / diff - 0.5);
    	printf("index %d error, value %f\n", index, data[index*2]);
    	data[index*2] += diff;
	}
	diff = result[1] - sum01;	
	if(diff<-delta || diff > delta)
	{
		printf("processor %d imaginary part error! %.15f %.15f diff %.10f\n", myRank, result[1], sum01, diff);
    	// fflush(stdout);
		//correction
		pos = data + 1;
    	R sum11 = 0;

    	for(i=1; i<=num; i++)
    	{
    		sum11 += i * pos[0];
    		pos += 2;
    	}
    	R diff2 = result[3] - sum11;
    	// int index = (int)(diff / diff2 - 0.5);
    	int index = (int)(diff2 / diff - 0.5);
    	printf("index %d error, value %f\n", index, data[index*2+1]);
    	data[index*2+1] += diff;
	}
	// if(myRank == 0) printf("verification done\n");
	return ;
}

static void naive_checksum_verification2(R * data, R delta, int size, R * cks1, R * cks2)
{
	R sum00 = 0, sum01 = 0;
	R * pos = data;
	int i;
	int num = size/2;
	for(i=0; i<num; i++)
	{
		sum00 += pos[0];// sum10 += cof[0] * pos[0];
		sum01 += pos[1];// sum11 += cof[0] * pos[1];
		pos += 2;
	}
	R diff = cks1[0] - sum00;
	if(diff<-delta || diff > delta)
	{
		printf("processor %d real part error! %.15f %.15f diff %.10f\n", myRank, cks1[0], sum00, diff);
    	// fflush(stdout);
		//corretion
    	pos = data;
    	R sum10 = 0;

    	for(i=1; i<=num; i++)
    	{
    		sum10 += i * pos[0];
    		pos += 2;
    	}
    	R diff2 = cks2[0] - sum10;
    	// int index = (int)(diff / diff2 - 0.5);
    	int index = (int)(diff2 / diff - 0.5);
    	printf("index %d error, value %f\n", index, data[index*2]);
    	data[index*2] += diff;
	}
	diff = cks1[1] - sum01;	
	if(diff<-delta || diff > delta)
	{
		printf("processor %d imaginary part error! %.15f %.15f diff %.10f\n", myRank, cks1[1], sum01, diff);
    	// fflush(stdout);
		//correction
		pos = data + 1;
    	R sum11 = 0;

    	for(i=1; i<=num; i++)
    	{
    		sum11 += i * pos[0];
    		pos += 2;
    	}
    	R diff2 = cks2[1] - sum11;
    	// int index = (int)(diff / diff2 - 0.5);
    	int index = (int)(diff2 / diff - 0.5);
    	printf("index %d error, value %f\n", index, data[index*2+1]);
    	data[index*2+1] += diff;
	}
	// if(myRank == 0) printf("verification done\n");
	return ;
}

static void transpose1_checksum_regeneration(R * data, R delta, R * rA, R * rAcoeff, int size, R * newCks1, R * newCks2, int index)
{
	R * cks = data + size;
	naive_checksum_verification(data, delta, size, cks);

	R * pos = data;
	R * cksPos = newCks1;
	R * cksPos2 = newCks2;
	register R c0 = rAcoeff[0];
	register R c1 = rAcoeff[1];
	register R rAc0 = rA[0];
	register R rAc1 = rA[1];
	register R temp0, temp1;
	int i;
	int num = size/2;
	for(i=0; i<num; i++)
	{
		temp0 = pos[0], temp1 = pos[1];
		cksPos[0] += rAc0*temp0 - rAc1*temp1;
		cksPos[1] += rAc1*temp0 + rAc0*temp1;
		cksPos2[0] += c0 * temp0;
		cksPos2[1] += c1 * temp1;

		pos += 2;
		cksPos += 2;
		cksPos2 += 2;		
	}
	// if(myRank == 0 && index<5) printf("cks data[%d]: %f %f\trA[%d]: %f %f\n", index, data[0], data[1], index, rA[0], rA[1]);
	return ;

}

static void transpose2_send_buffer_generation(R * send, int size, R * outCks1, R * outCks2)
{
	R * sb = send + size;
	sb[0] = outCks1[0];
	sb[1] = outCks1[1];
	sb[2] = outCks2[0];
	sb[3] = outCks2[1];
	return;	
}

static void transpose2_do_twiddle(triggen *t, INT ir, INT ms, INT me, INT vn, R *xr, R *xi)
{
     void (*rotate)(triggen *, INT, R, R, R *) = t->rotate;
     INT im, iv;
     for (im = ms; im < me; ++im)
     {
	  for (iv = 0; iv < vn; ++iv) {
	       /* TODO: modify/inline rotate function
		  so that it can do whole vn vector at once? */
	       R c[2];

	       rotate(t, ir * im, *xr, *xi, c);
	       *xr = c[0]; *xi = c[1];
	       xr += 2; xi += 2;
	  }
    }
}

// note: send to itself does not call this function
static void transpose2_recv_buffer_verification_and_regeneration(R * recv, R delta, R * rA, R * rAcoeff, int size, int radix, R * newCks1, R * newCks2, int pe)
{
	R * cks = recv + size;
	naive_checksum_verification(recv, delta, size, cks);
	// do twiddle after verifying data
	{
     INT ms, me, ir, vn;
     triggen *t;
  	 const dftr1P * myPlan = (const dftr1P *) dft_rank1_plan;
	 t = myPlan->t;
	 ir = myPlan->xmin;
	 vn = myPlan->vn;
	 int num = size / 2;
	 ms = pe * num;
	 me = (pe+1) * num;
	 // printf("t ir vn: %d %d %d\n", t, ir ,vn);
	 // printf("recv buffer size %d\n", size);
	 // fflush(stdout);

     clock_t start, end;
     double eTime;
     int uSleepTime;

     start = clock();

	 transpose2_do_twiddle(t, ir, ms, me, vn, recv, recv+1);

	 end = clock();
	 eTime = ((double)(end - start)) / CLOCKS_PER_SEC;
	 uSleepTime = (int) (eTime * 3*1e5);
	 usleep(uSleepTime);	 
	 // printf("transpose done\n");
	 // fflush(stdout);
	}

	R * pos = recv;
	R * cksPos;
	R * cksPos2;

	register R c0;
	register R c1;
	register R rAc0;
	register R rAc1;
	register R temp0, temp1;
	int i, j;
	int num = size / 2 / radix; // num of radix in a tranpose block
	int index = num*2*pe;
	for(i=0; i<num; i++)
	{
		c0 = rAcoeff[index];
		c1 = rAcoeff[index+1];
		rAc0 = rA[index];
		rAc1 = rA[index+1];
		cksPos = newCks1;
		cksPos2 = newCks2;
		for(j=0; j<radix; j++)
		{
			// if(myRank == 0 && j == 1)
			// {
			// 	printf("after transpose, %d: %f %f\n", num*pe+i, pos[0], pos[1]);
			// }

	        temp0 = pos[0], temp1 = pos[1];
			cksPos[0] += rAc0*temp0 - rAc1*temp1;
			cksPos[1] += rAc1*temp0 + rAc0*temp1;
			cksPos2[0] += c0 * temp0;
			cksPos2[1] += c1 * temp1;						
			pos += 2;
			cksPos += 2;
			cksPos2 += 2;		
		}
		index += 2;
	}
	return;
}

static void transpose3_send_buffer_init(R * send, int size, R * outCks1, R * outCks2)
{
	R * sb = send + size;
	sb[0] = outCks1[0];
	sb[1] = outCks1[1];
	sb[2] = outCks2[0];
	sb[3] = outCks2[1];
	return;	
}
static void transpose3_checksum_verification(R * recv, R delta, int size)
{
	naive_checksum_verification(recv, delta, size, recv + size);
	return;
}

static void transpose_chunks(int *sched, int n_pes, int my_pe,
			     INT *sbs, INT *sbo, INT *rbs, INT *rbo,
			     MPI_Comm comm,
			     R *I, R *O)
{
     if (sched) {
	  int i;
	  MPI_Status status;
	  MPI_Status status2;

	  /* TODO: explore non-synchronous send/recv? */

	  if (I == O) {
	       // R *buf = (R*) MALLOC(sizeof(R) * sbs[0], BUFFERS);

	       MPI_Request request;
	       MPI_Request request2;

	       if(firstTranspose)
	       {

		       // R *buf = (R*) MALLOC(sizeof(R) * ((int)sbs[0]), BUFFERS);

		       R * sendBuf = (R*) MALLOC(sizeof(R) * ((int)sbs[0] + 4), BUFFERS);
		       R * recvBuf = (R*) MALLOC(sizeof(R) * ((int)sbs[0] + 4), BUFFERS);

		       R * sendBuf2 = (R*) MALLOC(sizeof(R) * ((int)sbs[0] + 4), BUFFERS);
		       R * recvBuf2 = (R*) MALLOC(sizeof(R) * ((int)sbs[0] + 4), BUFFERS);		       

		    //    printf("processor%d buf size: %d\n", my_pe, sbs[0]);
		       int vl = sbs[0]; // number of fft * 2, = fftSize/pNum * 2;
		       // set delta
		       R delta1 = 1e-2;

		       int m = pNum;
		       checksum1 = (R *) calloc(vl, sizeof(R));
		       checksum2 = (R *) calloc(vl, sizeof(R));
			   
			   rA = (R *) malloc((m+2)*2*sizeof(R));
		       checksumVectorRAGeneration(m, rA);
		       rAcoeff = (R *) malloc(m*2*sizeof(R));
		       coefficientGeneration(m, rA, rAcoeff);
			   
	       	   // checksum generation
	       		R * initChecksum1 = (R *) malloc(2*pNum*sizeof(R));
	       		R * initChecksum2 = (R *) malloc(2*pNum*sizeof(R));
	       		{
	       			int i, j;
	       			int size = sbs[0] / 2;
	       			R * data = I;
	       			R * pos1 = initChecksum1;
	       			R * pos2 = initChecksum2;

	       			register R sum00, sum01, sum10, sum11;
		       		for(i=0; i<pNum; i++)
		       		{
		       			sum00 = 0, sum01 = 0, sum10 = 0, sum11 = 0;
		       			for(j=1; j<size+1; j++)
		       			{
			       			sum00 += data[0];
			       			sum01 += data[1];
			       			sum10 += j * data[0];
			       			sum11 += j * data[1];
			       			data += 2;
		       			}
		       			pos1[0] = sum00, pos1[1] = sum01;
		       			pos2[0] = sum10, pos2[1] = sum11;
		       			pos1 += 2, pos2 += 2;
		       		}

	       		}

	 			// inject error
	 			// if(myRank == 18)
	 			// {
	 			// 	I[374573] += 635;
	 			// }

	       		// if(myRank == 0) printf("init checksum done, pNum: %d\n", pNum);

		       int pe = sched[0];
		       int next_pe = sched[1];
			   int next_pe_size = (int) sbs[next_pe];
			   int last_pe;
			   int last_pe_size;
			   int pe_size;

		       // if(my_pe == 0) printf("pe: %d\nnext_pe: %d\nnext_pe_size: %d\n", pe, next_pe, next_pe_size);
		       // fflush(stdout);
		       // i = 0, always the communicate with itself
			    if (my_pe == pe) {
				 if (rbo[pe] != sbo[pe])
				 {
				      memmove(O + rbo[pe], O + sbo[pe],
					      sbs[pe] * sizeof(R));			      
				 }
				 // calculate new checksums (regeneration)
				 {
				 	pe_size = rbs[pe];
	       			{
						R * pos = O + rbo[pe];
						R result[4];
						result[0] = initChecksum1[2*pe], result[1] = initChecksum1[2*pe+1];
						result[2] = initChecksum2[2*pe], result[3] = initChecksum2[2*pe+1];
						naive_checksum_verification(pos, delta1, pe_size, result);
	       			}
		  			R * pos = O + rbo[pe];
					R * cksPos = checksum1;
					R * cksPos2 = checksum2;
					register R c0 = rAcoeff[2*pe];
					register R c1 = rAcoeff[2*pe+1];
					register R rAc0 = rA[2*pe];
					register R rAc1 = rA[2*pe+1];
					register R temp0, temp1;
					int i;
					for(i=0; i<pe_size/2; i++)
					{
	  			        temp0 = pos[0], temp1 = pos[1];
						cksPos[0] += rAc0*temp0 - rAc1*temp1;
						cksPos[1] += rAc1*temp0 + rAc0*temp1;
						cksPos2[0] += c0 * temp0;
						cksPos2[1] += c1 * temp1;						

						pos += 2;
						cksPos += 2;
						cksPos2 += 2;		
					}
				}
				  // calculate send buffer to next_pe
  				 memcpy(sendBuf, O+sbo[next_pe], next_pe_size * sizeof(R));
  				 transpose1_send_buffer_init(sendBuf, next_pe_size, initChecksum1+2*next_pe, initChecksum2+2*next_pe);

			    }
			    else
			    {
			    	printf("processor %d did not communicate with itself first but with %d, abort\n", my_pe, pe);
			    	exit(1);		    	
			    }

			    // consider my_pe!=pe in the first communication 
			   for (i = 1; i < n_pes-1; i+=2) {
			    	
			    	pe = sched[i];
				 	MPI_Isend(sendBuf, (int) (sbs[pe]) + 4, FFTW_MPI_TYPE, pe, (my_pe * n_pes + pe) & 0xffff, MPI_COMM_WORLD, &request);
				 	MPI_Irecv(recvBuf, (int) (rbs[pe]) + 4, FFTW_MPI_TYPE, pe, (pe * n_pes + my_pe) & 0xffff, MPI_COMM_WORLD, &request2);
			 		// verify recvBuf2 
			 		if(i>1)
			 		{
					 	last_pe = sched[i-1];
				 		last_pe_size = (int) rbs[last_pe];
				 		transpose1_checksum_regeneration(recvBuf2, delta1, rA+2*last_pe, rAcoeff+2*last_pe, last_pe_size, checksum1, checksum2, last_pe);
				 		memcpy(O+rbo[last_pe], recvBuf2, last_pe_size * sizeof(R));
			 		}				
		 			// generate sendBuf2
			 		next_pe = sched[i+1];
		 			next_pe_size = (int) sbs[next_pe];
		 			memcpy(sendBuf2, O+sbo[next_pe], next_pe_size * sizeof(R));
		 			transpose1_send_buffer_init(sendBuf2, next_pe_size, initChecksum1+2*next_pe, initChecksum2+2*next_pe);
				 	MPI_Wait(&request, &status);
				 	MPI_Wait(&request2, &status2);

				 	last_pe = pe;
				 	pe = next_pe;
				 	next_pe = sched[i+2];
				 	MPI_Isend(sendBuf2, (int) (sbs[pe]) + 4, FFTW_MPI_TYPE, pe, (my_pe * n_pes + pe) & 0xffff, MPI_COMM_WORLD, &request);
				 	MPI_Irecv(recvBuf2, (int) (rbs[pe]) + 4, FFTW_MPI_TYPE, pe, (pe * n_pes + my_pe) & 0xffff, MPI_COMM_WORLD, &request2);
			 		// verify recvBuf 
		 			// if(i == 9)
		 			// {
		 			// 	recvBuf[8459] += 722;
		 			// }				 	

			 		last_pe_size = (int) rbs[last_pe];
			 		transpose1_checksum_regeneration(recvBuf, delta1, rA+2*last_pe, rAcoeff+2*last_pe, last_pe_size, checksum1, checksum2, last_pe);
			 		memcpy(O+rbo[last_pe], recvBuf, last_pe_size * sizeof(R));
		 			// generate sendBuf
		 			next_pe_size = (int) sbs[next_pe];
		 			memcpy(sendBuf, O+sbo[next_pe], next_pe_size * sizeof(R));
		 			transpose1_send_buffer_init(sendBuf, next_pe_size, initChecksum1+2*next_pe, initChecksum2+2*next_pe);
				 	MPI_Wait(&request, &status);
				 	MPI_Wait(&request2, &status2);				 	

		       }// end for
		        pe = sched[n_pes-1];	        		       
			 	MPI_Isend(sendBuf, (int) (sbs[pe]) + 4, FFTW_MPI_TYPE, pe, (my_pe * n_pes + pe) & 0xffff, MPI_COMM_WORLD, &request);
			 	MPI_Irecv(recvBuf, (int) (rbs[pe]) + 4, FFTW_MPI_TYPE, pe, (pe * n_pes + my_pe) & 0xffff, MPI_COMM_WORLD, &request2);
			 	// verify last_pe
			 	last_pe = sched[n_pes-2];
		 		last_pe_size = (int) rbs[last_pe];
		 		transpose1_checksum_regeneration(recvBuf2, delta1, rA+2*last_pe, rAcoeff+2*last_pe, last_pe_size, checksum1, checksum2, last_pe);
		 		memcpy(O+rbo[last_pe], recvBuf2, last_pe_size * sizeof(R));
			 	MPI_Wait(&request, &status);
			 	MPI_Wait(&request2, &status2);
			 	last_pe = pe;
		 		last_pe_size = (int) rbs[last_pe];
		 		transpose1_checksum_regeneration(recvBuf, delta1, rA+2*last_pe, rAcoeff+2*last_pe, last_pe_size, checksum1, checksum2, last_pe);
		 		memcpy(O+rbo[last_pe], recvBuf, last_pe_size * sizeof(R));			 	

		        // X(ifree)(buf);
				X(ifree)(sendBuf);
				X(ifree)(recvBuf);
				X(ifree)(sendBuf2);
				X(ifree)(recvBuf2);
				free(initChecksum1);
				free(initChecksum2);
		       // free(buf);
		   }// end if firstTranspose
		   else if(secondTranspose)
		   {
		       R * sendBuf = (R*) MALLOC(sizeof(R) * ((int)sbs[0] + 4), BUFFERS);
		       R * recvBuf = (R*) MALLOC(sizeof(R) * ((int)sbs[0] + 4), BUFFERS);

		       R * sendBuf2 = (R*) MALLOC(sizeof(R) * ((int)sbs[0] + 4), BUFFERS);
		       R * recvBuf2 = (R*) MALLOC(sizeof(R) * ((int)sbs[0] + 4), BUFFERS);		       


		       // int vl = sbs[0]; // number of fft * 2 in fft1, = fftSize/pNum * 2;
	       		// if(myRank == 0) printf("secondTranspose radix %d myRadix %d\n", fft2Radix, myRadix);
	       		// fflush(stdout);

		       int radix = fft2Radix;
		       int m = myRadix / radix;
		    //    if(myRank == 0) printf("bufSize %d m %d fft2Radix %d\n", sbs[0], m, fft2Radix);
			   // fflush(stdout);
		       // set delta;
		       R delta2 = 1e-2;

		       fft2checksum1 = (R *) calloc(2*radix, sizeof(R));
		       fft2checksum2 = (R *) calloc(2*radix, sizeof(R));

		       rA = (R *) malloc((m+2)*2*sizeof(R));
		       checksumVectorRAGeneration(m, rA);
		       rAcoeff = (R *) malloc(m*2*sizeof(R));
		       coefficientGeneration(m, rA, rAcoeff);
			   

		       int pe = sched[0];
		       int next_pe = sched[1];
			   int next_pe_size = (int) sbs[next_pe];
			   int last_pe;
			   int last_pe_size;
			   int pe_size;

			    if (my_pe == pe) {
				 if (rbo[pe] != sbo[pe])
				 {
				      memmove(O + rbo[pe], O + sbo[pe],
					      sbs[pe] * sizeof(R));			      
				 }
				 // calculate new checksums (regeneration)
				 {
				 	{
						naive_checksum_verification2(O + rbo[pe], delta2, rbs[pe], outChecksum1+2*pe, outChecksum2+2*pe);
				 	}
					{
					 int size = rbs[pe];
				     INT ms, me, ir, vn;
				     triggen *t;
				  	 const dftr1P * myPlan = (const dftr1P *) dft_rank1_plan;
					 t = myPlan->t;
					 ir = myPlan->xmin;
					 vn = myPlan->vn;
					 int num = size / 2;
					 ms = pe * num;
					 me = (pe+1) * num;
					 // printf("t ir vn: %d %d %d\n", t, ir ,vn);
					 // printf("recv buffer size %d\n", size);
					 // fflush(stdout);
					 transpose2_do_twiddle(t, ir, ms, me, vn, O+rbo[pe], O+rbo[pe]+1);
					 // printf("transpose done\n");
					 // fflush(stdout);
					}				 	
				 	pe_size = rbs[pe];
		  			R * pos = O + rbo[pe];
					R * cksPos;
					R * cksPos2;
					register R c0;
					register R c1;
					register R rAc0;
					register R rAc1;
					register R temp0, temp1;
					int i, j;
					int num = pe_size / 2 / radix; // num of radix in a tranpose block
					int index = num*2*pe;
					R sum00 = 0, sum01 = 0;
					for(i=0; i<num; i++)
					{
						c0 = rAcoeff[index];
						c1 = rAcoeff[index+1];
						rAc0 = rA[index];
						rAc1 = rA[index+1];
						cksPos = fft2checksum1;
						cksPos2 = fft2checksum2;
						for(j=0; j<radix; j++)
						{
		  			        temp0 = pos[0], temp1 = pos[1];
							cksPos[0] += rAc0*temp0 - rAc1*temp1;
							cksPos[1] += rAc1*temp0 + rAc0*temp1;
							cksPos2[0] += c0 * temp0;
							cksPos2[1] += c1 * temp1;						
							sum00 += temp0;
							sum01 += temp1;

							pos += 2;
							cksPos += 2;
							cksPos2 += 2;		
						}
						index += 2;
					}
				}
				  // calculate send buffer to next_pe
  				 memcpy(sendBuf, O+sbo[next_pe], next_pe_size * sizeof(R));
				 transpose2_send_buffer_generation(sendBuf, next_pe_size, outChecksum1+2*next_pe, outChecksum2+2*next_pe);

			    }
			    else
			    {
			    	printf("processor %d did not communicate with itself first but with %d, abort\n", my_pe, pe);
			    	exit(1);		    	
			    }

			    // consider my_pe!=pe in the first communication 
			   for (i = 1; i < n_pes-1; i+=2) {
			    	
			    	pe = sched[i];
				 	MPI_Isend(sendBuf, (int) (sbs[pe]) + 4, FFTW_MPI_TYPE, pe, (my_pe * n_pes + pe) & 0xffff, MPI_COMM_WORLD, &request);
				 	MPI_Irecv(recvBuf, (int) (rbs[pe]) + 4, FFTW_MPI_TYPE, pe, (pe * n_pes + my_pe) & 0xffff, MPI_COMM_WORLD, &request2);
			 		// verify recvBuf2 
			 		if(i>1)
			 		{
			 			// inject error
			 			// if(i == 9)
			 			// {
			 			// 	recvBuf2[37452] += 6905;
			 			// }
					 	last_pe = sched[i-1];
				 		last_pe_size = (int) rbs[last_pe];
				 		transpose2_recv_buffer_verification_and_regeneration(recvBuf2, delta2, rA, rAcoeff, last_pe_size, radix, fft2checksum1, fft2checksum2, last_pe);
				 		memcpy(O+rbo[last_pe], recvBuf2, last_pe_size * sizeof(R));
			 		}				
		 			// generate sendBuf2
			 		next_pe = sched[i+1];
		 			next_pe_size = (int) sbs[next_pe];
		 			memcpy(sendBuf2, O+sbo[next_pe], next_pe_size * sizeof(R));
					transpose2_send_buffer_generation(sendBuf2, next_pe_size, outChecksum1+2*next_pe, outChecksum2+2*next_pe);
				 	MPI_Wait(&request, &status);
				 	MPI_Wait(&request2, &status2);

				 	last_pe = pe;
				 	pe = next_pe;
				 	next_pe = sched[i+2];
				 	MPI_Isend(sendBuf2, (int) (sbs[pe]) + 4, FFTW_MPI_TYPE, pe, (my_pe * n_pes + pe) & 0xffff, MPI_COMM_WORLD, &request);
				 	MPI_Irecv(recvBuf2, (int) (rbs[pe]) + 4, FFTW_MPI_TYPE, pe, (pe * n_pes + my_pe) & 0xffff, MPI_COMM_WORLD, &request2);
			 		// verify recvBuf 
			 		last_pe_size = (int) rbs[last_pe];
			 		transpose2_recv_buffer_verification_and_regeneration(recvBuf, delta2, rA, rAcoeff, last_pe_size, radix, fft2checksum1, fft2checksum2, last_pe);
			 		memcpy(O+rbo[last_pe], recvBuf, last_pe_size * sizeof(R));
		 			// generate sendBuf
		 			next_pe_size = (int) sbs[next_pe];
		 			memcpy(sendBuf, O+sbo[next_pe], next_pe_size * sizeof(R));
					transpose2_send_buffer_generation(sendBuf, next_pe_size, outChecksum1+2*next_pe, outChecksum2+2*next_pe);
				 	MPI_Wait(&request, &status);
				 	MPI_Wait(&request2, &status2);				 	

		       }// end for
		        pe = sched[n_pes-1];	        		       
			 	MPI_Isend(sendBuf, (int) (sbs[pe]) + 4, FFTW_MPI_TYPE, pe, (my_pe * n_pes + pe) & 0xffff, MPI_COMM_WORLD, &request);
			 	MPI_Irecv(recvBuf, (int) (rbs[pe]) + 4, FFTW_MPI_TYPE, pe, (pe * n_pes + my_pe) & 0xffff, MPI_COMM_WORLD, &request2);
			 	// verify last_pe
			 	last_pe = sched[n_pes-2];
		 		last_pe_size = (int) rbs[last_pe];
		 		transpose2_recv_buffer_verification_and_regeneration(recvBuf2, delta2, rA, rAcoeff, last_pe_size, radix, fft2checksum1, fft2checksum2, last_pe);
		 		memcpy(O+rbo[last_pe], recvBuf2, last_pe_size * sizeof(R));
			 	MPI_Wait(&request, &status);
			 	MPI_Wait(&request2, &status2);
			 	last_pe = pe;
		 		last_pe_size = (int) rbs[last_pe];
		 		transpose2_recv_buffer_verification_and_regeneration(recvBuf, delta2, rA, rAcoeff, last_pe_size, radix, fft2checksum1, fft2checksum2, last_pe);
		 		memcpy(O+rbo[last_pe], recvBuf, last_pe_size * sizeof(R));			 	

		        // X(ifree)(buf);
				X(ifree)(sendBuf);
				X(ifree)(recvBuf);
				X(ifree)(sendBuf2);
				X(ifree)(recvBuf2);
				free(outChecksum1);
				free(outChecksum2);
		       // free(buf);
		   }// end secondTranspose
		   else 	// the last transpose, rearrange data position
		   {
		     // printf("processor %d third transpose\n", myRank);
		     // fflush(stdout);

		   	   int vl = (int)sbs[0];
		       R * sendBuf = (R*) MALLOC(sizeof(R) * (vl + 4), BUFFERS);
		       R * recvBuf = (R*) MALLOC(sizeof(R) * (vl + 4), BUFFERS);

		       R * sendBuf2 = (R*) MALLOC(sizeof(R) * (vl + 4), BUFFERS);
		       R * recvBuf2 = (R*) MALLOC(sizeof(R) * (vl + 4), BUFFERS);	

		   	   if(myRadix == 0) printf("processor %d third transpose\n", myRank);
				fflush(stdout);

		       int pe = sched[0];
		       int next_pe = sched[1];
			   int next_pe_size = (int) sbs[next_pe];
			   int last_pe;
			   int last_pe_size;
			   int pe_size;

		       // if(my_pe == 0) printf("pe: %d\nnext_pe: %d\nnext_pe_size: %d\n", pe, next_pe, next_pe_size);
		       // fflush(stdout);
		       // i = 0, always the communicate with itself
			    if (my_pe == pe) {
				 if (rbo[pe] != sbo[pe])
				 {
				      memmove(O + rbo[pe], O + sbo[pe],
					      sbs[pe] * sizeof(R));			      
				 }
				  // calculate send buffer to next_pe
  				 memcpy(sendBuf, O+sbo[next_pe], next_pe_size * sizeof(R));
  				 transpose3_send_buffer_init(sendBuf, next_pe_size, outputChecksum1+2*next_pe, outputChecksum2+2*next_pe);

			    }
			    else
			    {
			    	printf("processor %d did not communicate with itself first but with %d, abort\n", my_pe, pe);
			    	exit(1);		    	
			    }

			    // consider my_pe!=pe in the first communication 
			   for (i = 1; i < n_pes-1; i+=2) {
			    	
			    	pe = sched[i];
				 	MPI_Isend(sendBuf, (int) (sbs[pe]) + 4, FFTW_MPI_TYPE, pe, (my_pe * n_pes + pe) & 0xffff, MPI_COMM_WORLD, &request);
				 	MPI_Irecv(recvBuf, (int) (rbs[pe]) + 4, FFTW_MPI_TYPE, pe, (pe * n_pes + my_pe) & 0xffff, MPI_COMM_WORLD, &request2);
			 		// verify recvBuf2 
			 		if(i>1)
			 		{
			 			// if(i == 21)
			 			// {
			 			// 	recvBuf2[2899] += 1748;
			 			// }	
					 	last_pe = sched[i-1];
				 		last_pe_size = (int) rbs[last_pe];
				 		// transpose3_checksum_verification(recvBuf2, delta3, last_pe_size);
				 		memcpy(O+rbo[last_pe], recvBuf2, last_pe_size * sizeof(R));
				 		outputChecksum1[2*last_pe] = recvBuf2[last_pe_size];
				 		outputChecksum1[2*last_pe+1] = recvBuf2[last_pe_size+1];
				 		outputChecksum2[2*last_pe] = recvBuf2[last_pe_size+2];
				 		outputChecksum2[2*last_pe+1] = recvBuf2[last_pe_size+3];

			 		}				
		 			// generate sendBuf2
			 		next_pe = sched[i+1];
		 			next_pe_size = (int) sbs[next_pe];
		 			memcpy(sendBuf2, O+sbo[next_pe], next_pe_size * sizeof(R));
					transpose3_send_buffer_init(sendBuf2, next_pe_size, outputChecksum1+2*next_pe, outputChecksum2+2*next_pe);

				 	MPI_Wait(&request, &status);
				 	MPI_Wait(&request2, &status2);

				 	last_pe = pe;
				 	pe = next_pe;
				 	next_pe = sched[i+2];
				 	MPI_Isend(sendBuf2, (int) (sbs[pe]) + 4, FFTW_MPI_TYPE, pe, (my_pe * n_pes + pe) & 0xffff, MPI_COMM_WORLD, &request);
				 	MPI_Irecv(recvBuf2, (int) (rbs[pe]) + 4, FFTW_MPI_TYPE, pe, (pe * n_pes + my_pe) & 0xffff, MPI_COMM_WORLD, &request2);
			 		// verify recvBuf 
			 		last_pe_size = (int) rbs[last_pe];
			 		// transpose3_checksum_verification(recvBuf, delta3, last_pe_size);
			 		memcpy(O+rbo[last_pe], recvBuf, last_pe_size * sizeof(R));
			 		outputChecksum1[2*last_pe] = recvBuf[last_pe_size];
			 		outputChecksum1[2*last_pe+1] = recvBuf[last_pe_size+1];
			 		outputChecksum2[2*last_pe] = recvBuf[last_pe_size+2];
			 		outputChecksum2[2*last_pe+1] = recvBuf[last_pe_size+3];

		 			// generate sendBuf
		 			next_pe_size = (int) sbs[next_pe];
		 			memcpy(sendBuf, O+sbo[next_pe], next_pe_size * sizeof(R));
					transpose3_send_buffer_init(sendBuf, next_pe_size, outputChecksum1+2*next_pe, outputChecksum2+2*next_pe);

				 	MPI_Wait(&request, &status);
				 	MPI_Wait(&request2, &status2);				 	

		       }// end for
		        pe = sched[n_pes-1];	        		       
			 	MPI_Isend(sendBuf, (int) (sbs[pe]) + 4, FFTW_MPI_TYPE, pe, (my_pe * n_pes + pe) & 0xffff, MPI_COMM_WORLD, &request);
			 	MPI_Irecv(recvBuf, (int) (rbs[pe]) + 4, FFTW_MPI_TYPE, pe, (pe * n_pes + my_pe) & 0xffff, MPI_COMM_WORLD, &request2);
			 	// verify last_pe
			 	last_pe = sched[n_pes-2];
		 		last_pe_size = (int) rbs[last_pe];
		 		// transpose3_checksum_verification(recvBuf2, delta3, last_pe_size);
		 		memcpy(O+rbo[last_pe], recvBuf2, last_pe_size * sizeof(R));
		 		outputChecksum1[2*last_pe] = recvBuf2[last_pe_size];
		 		outputChecksum1[2*last_pe+1] = recvBuf2[last_pe_size+1];
		 		outputChecksum2[2*last_pe] = recvBuf2[last_pe_size+2];
		 		outputChecksum2[2*last_pe+1] = recvBuf2[last_pe_size+3];

			 	MPI_Wait(&request, &status);
			 	MPI_Wait(&request2, &status2);
			 	last_pe = pe;
		 		last_pe_size = (int) rbs[last_pe];
		 		// transpose3_checksum_verification(recvBuf, delta3, last_pe_size);
		 		memcpy(O+rbo[last_pe], recvBuf, last_pe_size * sizeof(R));			 	
		 		outputChecksum1[2*last_pe] = recvBuf[last_pe_size];
		 		outputChecksum1[2*last_pe+1] = recvBuf[last_pe_size+1];
		 		outputChecksum2[2*last_pe] = recvBuf[last_pe_size+2];
		 		outputChecksum2[2*last_pe+1] = recvBuf[last_pe_size+3];

		 		// can only verify after receiving all data
			   	//set delta
		   		R delta3 = 1e-1;
		   		{
		   			R * cksPos1 = outputChecksum1;
		   			R * cksPos2 = outputChecksum2;
			   		int size;
			   		for(i=0; i<n_pes; i++)
			   		{
			   			size = rbs[i];
			   			naive_checksum_verification2(O+rbo[i], delta3, size, cksPos1, cksPos2);
			   			cksPos1 += 2, cksPos2 += 2;
			   		}
		   		}

		        // X(ifree)(buf);
				X(ifree)(sendBuf);
				X(ifree)(recvBuf);
				X(ifree)(sendBuf2);
				X(ifree)(recvBuf2);
				free(outputChecksum1);
				free(outputChecksum2);		       


		   }
	  }
	  else { /* I != O */
	       for (i = 0; i < n_pes; ++i) {
		    int pe = sched[i];
		    if (my_pe == pe)
			 memcpy(O + rbo[pe], I + sbo[pe], sbs[pe] * sizeof(R));
		    else
			 MPI_Sendrecv(I + sbo[pe], (int) (sbs[pe]),
				      FFTW_MPI_TYPE,
				      pe, (my_pe * n_pes + pe) & 0xffff,
				      O + rbo[pe], (int) (rbs[pe]),
				      FFTW_MPI_TYPE,
				      pe, (pe * n_pes + my_pe) & 0xffff,
				      comm, &status);
	       }
	  }
     }
}

static void apply(const plan *ego_, R *I, R *O)
{
     const P *ego = (const P *) ego_;
     plan_rdft *cld1, *cld2, *cld2rest, *cld3;

     /* transpose locally to get contiguous chunks */
     cld1 = (plan_rdft *) ego->cld1;
     if (cld1) {
	  cld1->apply(ego->cld1, I, O);
	  
	  if (ego->preserve_input) I = O;

	  /* transpose chunks globally */
	  transpose_chunks(ego->sched, ego->n_pes, ego->my_pe,
			   ego->send_block_sizes, ego->send_block_offsets,
			   ego->recv_block_sizes, ego->recv_block_offsets,
			   ego->comm, O, I);
     }
     else if (ego->preserve_input) {
	  /* transpose chunks globally */
	  transpose_chunks(ego->sched, ego->n_pes, ego->my_pe,
			   ego->send_block_sizes, ego->send_block_offsets,
			   ego->recv_block_sizes, ego->recv_block_offsets,
			   ego->comm, I, O);

	  I = O;
     }
     else {
	  /* transpose chunks globally */
	  transpose_chunks(ego->sched, ego->n_pes, ego->my_pe,
			   ego->send_block_sizes, ego->send_block_offsets,
			   ego->recv_block_sizes, ego->recv_block_offsets,
			   ego->comm, I, I);
     }

     /* transpose locally, again, to get ordinary row-major;
	this may take two transposes if the block sizes are unequal
	(3 subplans, two of which operate on disjoint data) */
     cld2 = (plan_rdft *) ego->cld2;
     cld2->apply(ego->cld2, I, O);
     cld2rest = (plan_rdft *) ego->cld2rest;
     if (cld2rest) {
	  cld2rest->apply(ego->cld2rest,
			  I + ego->rest_Ioff, O + ego->rest_Ooff);
	  cld3 = (plan_rdft *) ego->cld3;
	  if (cld3)
	       cld3->apply(ego->cld3, O, O);
	  /* else TRANSPOSED_OUT is true and user wants O transposed */
     }
}

static int applicable(const S *ego, const problem *p_,
		      const planner *plnr)
{
     const problem_mpi_transpose *p = (const problem_mpi_transpose *) p_;
     /* Note: this is *not* UGLY for out-of-place, destroy-input plans;
	the planner often prefers transpose-pairwise to transpose-alltoall,
	at least with LAM MPI on my machine. */
     return (1
	     && (!ego->preserve_input || (!NO_DESTROY_INPUTP(plnr)
					  && p->I != p->O))
	     && ONLY_TRANSPOSEDP(p->flags));
}

static void awake(plan *ego_, enum wakefulness wakefulness)
{
     P *ego = (P *) ego_;
     X(plan_awake)(ego->cld1, wakefulness);
     X(plan_awake)(ego->cld2, wakefulness);
     X(plan_awake)(ego->cld2rest, wakefulness);
     X(plan_awake)(ego->cld3, wakefulness);
}

static void destroy(plan *ego_)
{
     P *ego = (P *) ego_;
     X(ifree0)(ego->sched);
     X(ifree0)(ego->send_block_sizes);
     MPI_Comm_free(&ego->comm);
     X(plan_destroy_internal)(ego->cld3);
     X(plan_destroy_internal)(ego->cld2rest);
     X(plan_destroy_internal)(ego->cld2);
     X(plan_destroy_internal)(ego->cld1);
}

static void print(const plan *ego_, printer *p)
{
     const P *ego = (const P *) ego_;
     p->print(p, "(mpi-transpose-pairwise%s%(%p%)%(%p%)%(%p%)%(%p%))", 
	      ego->preserve_input==2 ?"/p":"",
	      ego->cld1, ego->cld2, ego->cld2rest, ego->cld3);
}

/* Given a process which_pe and a number of processes npes, fills
   the array sched[npes] with a sequence of processes to communicate
   with for a deadlock-free, optimum-overlap all-to-all communication.
   (All processes must call this routine to get their own schedules.)
   The schedule can be re-ordered arbitrarily as long as all processes
   apply the same permutation to their schedules.

   The algorithm here is based upon the one described in:
       J. A. M. Schreuder, "Constructing timetables for sport
       competitions," Mathematical Programming Study 13, pp. 58-67 (1980). 
   In a sport competition, you have N teams and want every team to
   play every other team in as short a time as possible (maximum overlap
   between games).  This timetabling problem is therefore identical
   to that of an all-to-all communications problem.  In our case, there
   is one wrinkle: as part of the schedule, the process must do
   some data transfer with itself (local data movement), analogous
   to a requirement that each team "play itself" in addition to other
   teams.  With this wrinkle, it turns out that an optimal timetable
   (N parallel games) can be constructed for any N, not just for even
   N as in the original problem described by Schreuder.
*/
static void fill1_comm_sched(int *sched, int which_pe, int npes)
{
     int pe, i, n, s = 0;
     A(which_pe >= 0 && which_pe < npes);
     if (npes % 2 == 0) {
	  n = npes;
	  sched[s++] = which_pe;
     }
     else
	  n = npes + 1;
     for (pe = 0; pe < n - 1; ++pe) {
	  if (npes % 2 == 0) {
	       if (pe == which_pe) sched[s++] = npes - 1;
	       else if (npes - 1 == which_pe) sched[s++] = pe;
	  }
	  else if (pe == which_pe) sched[s++] = pe;

	  if (pe != which_pe && which_pe < n - 1) {
	       i = (pe - which_pe + (n - 1)) % (n - 1);
	       if (i < n/2)
		    sched[s++] = (pe + i) % (n - 1);
	       
	       i = (which_pe - pe + (n - 1)) % (n - 1);
	       if (i < n/2)
		    sched[s++] = (pe - i + (n - 1)) % (n - 1);
	  }
     }
     A(s == npes);
}

/* Sort the communication schedule sched for npes so that the schedule
   on process sortpe is ascending or descending (!ascending).  This is
   necessary to allow in-place transposes when the problem does not
   divide equally among the processes.  In this case there is one
   process where the incoming blocks are bigger/smaller than the
   outgoing blocks and thus have to be received in
   descending/ascending order, respectively, to avoid overwriting data
   before it is sent. */
static void sort1_comm_sched(int *sched, int npes, int sortpe, int ascending)
{
     int *sortsched, i;
     sortsched = (int *) MALLOC(npes * sizeof(int) * 2, OTHER);
     fill1_comm_sched(sortsched, sortpe, npes);
     if (ascending)
	  for (i = 0; i < npes; ++i)
	       sortsched[npes + sortsched[i]] = sched[i];
     else
	  for (i = 0; i < npes; ++i)
	       sortsched[2*npes - 1 - sortsched[i]] = sched[i];
     for (i = 0; i < npes; ++i)
	  sched[i] = sortsched[npes + i];
     X(ifree)(sortsched);
}

/* make the plans to do the post-MPI transpositions (shared with
   transpose-alltoall) */
int XM(mkplans_posttranspose)(const problem_mpi_transpose *p, planner *plnr,
			      R *I, R *O, int my_pe,
			      plan **cld2, plan **cld2rest, plan **cld3,
			      INT *rest_Ioff, INT *rest_Ooff)
{
     INT vn = p->vn;
     INT b = p->block;
     INT bt = XM(block)(p->ny, p->tblock, my_pe);
     INT nxb = p->nx / b; /* number of equal-sized blocks */
     INT nxr = p->nx - nxb * b; /* leftover rows after equal blocks */

     *cld2 = *cld2rest = *cld3 = NULL;
     *rest_Ioff = *rest_Ooff = 0;

     if (!(p->flags & TRANSPOSED_OUT) && (nxr == 0 || I != O)) {
	  INT nx = p->nx * vn;
	  b *= vn;
	  *cld2 = X(mkplan_f_d)(plnr, 
				X(mkproblem_rdft_0_d)(X(mktensor_3d)
						      (nxb, bt * b, b,
						       bt, b, nx,
						       b, 1, 1),
						      I, O),
				0, 0, NO_SLOW);
	  if (!*cld2) goto nada;

	  if (nxr > 0) {
	       *rest_Ioff = nxb * bt * b;
	       *rest_Ooff = nxb * b;
	       b = nxr * vn;
	       *cld2rest = X(mkplan_f_d)(plnr,
					 X(mkproblem_rdft_0_d)(X(mktensor_2d)
							       (bt, b, nx,
								b, 1, 1),
							       I + *rest_Ioff,
							       O + *rest_Ooff),
                                        0, 0, NO_SLOW);
               if (!*cld2rest) goto nada;
	  }
     }
     else {
	  *cld2 = X(mkplan_f_d)(plnr,
				X(mkproblem_rdft_0_d)(
				     X(mktensor_4d)
				     (nxb, bt * b * vn, bt * b * vn,
				      bt, b * vn, vn,
				      b, vn, bt * vn,
				      vn, 1, 1),
				     I, O),
				0, 0, NO_SLOW);
	  if (!*cld2) goto nada;

	  *rest_Ioff = *rest_Ooff = nxb * bt * b * vn;
	  *cld2rest = X(mkplan_f_d)(plnr,
				    X(mkproblem_rdft_0_d)(
					 X(mktensor_3d)
					 (bt, nxr * vn, vn,
					  nxr, vn, bt * vn,
					  vn, 1, 1),
					 I + *rest_Ioff, O + *rest_Ooff),
				    0, 0, NO_SLOW);
	  if (!*cld2rest) goto nada;

	  if (!(p->flags & TRANSPOSED_OUT)) {
	       *cld3 = X(mkplan_f_d)(plnr,
				     X(mkproblem_rdft_0_d)(
					  X(mktensor_3d)
					  (p->nx, bt * vn, vn,
					   bt, vn, p->nx * vn,
					   vn, 1, 1),
					  O, O),
				     0, 0, NO_SLOW);
	       if (!*cld3) goto nada;
	  }
     }

     return 1;

nada:
     X(plan_destroy_internal)(*cld3);
     X(plan_destroy_internal)(*cld2rest);
     X(plan_destroy_internal)(*cld2);
     *cld2 = *cld2rest = *cld3 = NULL;
     return 0;
}

static plan *mkplan(const solver *ego_, const problem *p_, planner *plnr)
{
     const S *ego = (const S *) ego_;
     const problem_mpi_transpose *p;
     P *pln;
     plan *cld1 = 0, *cld2 = 0, *cld2rest = 0, *cld3 = 0;
     INT b, bt, vn, rest_Ioff, rest_Ooff;
     INT *sbs, *sbo, *rbs, *rbo;
     int pe, my_pe, n_pes, sort_pe = -1, ascending = 1;
     R *I, *O;
     static const plan_adt padt = {
          XM(transpose_solve), awake, print, destroy
     };

     UNUSED(ego);

     if (!applicable(ego, p_, plnr))
          return (plan *) 0;

     p = (const problem_mpi_transpose *) p_;
     vn = p->vn;
     I = p->I; O = p->O;

     MPI_Comm_rank(p->comm, &my_pe);
     MPI_Comm_size(p->comm, &n_pes);

     b = XM(block)(p->nx, p->block, my_pe);
     
     if (!(p->flags & TRANSPOSED_IN)) { /* b x ny x vn -> ny x b x vn */
	  cld1 = X(mkplan_f_d)(plnr, 
			       X(mkproblem_rdft_0_d)(X(mktensor_3d)
						     (b, p->ny * vn, vn,
						      p->ny, vn, b * vn,
						      vn, 1, 1),
						     I, O),
			       0, 0, NO_SLOW);
	  if (XM(any_true)(!cld1, p->comm)) goto nada;
     }
     if (ego->preserve_input || NO_DESTROY_INPUTP(plnr)) I = O;

     if (XM(any_true)(!XM(mkplans_posttranspose)(p, plnr, I, O, my_pe,
						 &cld2, &cld2rest, &cld3,
						 &rest_Ioff, &rest_Ooff),
		      p->comm)) goto nada;

     pln = MKPLAN_MPI_TRANSPOSE(P, &padt, apply);

     pln->cld1 = cld1;
     pln->cld2 = cld2;
     pln->cld2rest = cld2rest;
     pln->rest_Ioff = rest_Ioff;
     pln->rest_Ooff = rest_Ooff;
     pln->cld3 = cld3;
     pln->preserve_input = ego->preserve_input ? 2 : NO_DESTROY_INPUTP(plnr);

     MPI_Comm_dup(p->comm, &pln->comm);

     n_pes = (int) X(imax)(XM(num_blocks)(p->nx, p->block),
			   XM(num_blocks)(p->ny, p->tblock));

     /* Compute sizes/offsets of blocks to exchange between processors */
     sbs = (INT *) MALLOC(4 * n_pes * sizeof(INT), PLANS);
     sbo = sbs + n_pes;
     rbs = sbo + n_pes;
     rbo = rbs + n_pes;
     b = XM(block)(p->nx, p->block, my_pe);
     bt = XM(block)(p->ny, p->tblock, my_pe);
     for (pe = 0; pe < n_pes; ++pe) {
	  INT db, dbt; /* destination block sizes */
	  db = XM(block)(p->nx, p->block, pe);
	  dbt = XM(block)(p->ny, p->tblock, pe);

	  sbs[pe] = b * dbt * vn;
	  sbo[pe] = pe * (b * p->tblock) * vn;
	  rbs[pe] = db * bt * vn;
	  rbo[pe] = pe * (p->block * bt) * vn;

	  if (db * dbt > 0 && db * p->tblock != p->block * dbt) {
	       A(sort_pe == -1); /* only one process should need sorting */
	       sort_pe = pe;
	       ascending = db * p->tblock > p->block * dbt;
	  }
     }
     pln->n_pes = n_pes;
     pln->my_pe = my_pe;
     pln->send_block_sizes = sbs;
     pln->send_block_offsets = sbo;
     pln->recv_block_sizes = rbs;
     pln->recv_block_offsets = rbo;

     if (my_pe >= n_pes) {
	  pln->sched = 0; /* this process is not doing anything */
     }
     else {
	  pln->sched = (int *) MALLOC(n_pes * sizeof(int), PLANS);
	  fill1_comm_sched(pln->sched, my_pe, n_pes);
	  if (sort_pe >= 0)
	       sort1_comm_sched(pln->sched, n_pes, sort_pe, ascending);
     }

     X(ops_zero)(&pln->super.super.ops);
     if (cld1) X(ops_add2)(&cld1->ops, &pln->super.super.ops);
     if (cld2) X(ops_add2)(&cld2->ops, &pln->super.super.ops);
     if (cld2rest) X(ops_add2)(&cld2rest->ops, &pln->super.super.ops);
     if (cld3) X(ops_add2)(&cld3->ops, &pln->super.super.ops);
     /* FIXME: should MPI operations be counted in "other" somehow? */

     return &(pln->super.super);

 nada:
     X(plan_destroy_internal)(cld3);
     X(plan_destroy_internal)(cld2rest);
     X(plan_destroy_internal)(cld2);
     X(plan_destroy_internal)(cld1);
     return (plan *) 0;
}

static solver *mksolver(int preserve_input)
{
     static const solver_adt sadt = { PROBLEM_MPI_TRANSPOSE, mkplan, 0 };
     S *slv = MKSOLVER(S, &sadt);
     slv->preserve_input = preserve_input;
     return &(slv->super);
}

void XM(transpose_pairwise_register)(planner *p)
{
     int preserve_input;
     for (preserve_input = 0; preserve_input <= 1; ++preserve_input)
	  REGISTER_SOLVER(p, mksolver(preserve_input));
}
