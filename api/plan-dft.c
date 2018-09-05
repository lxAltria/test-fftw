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

#include "api.h"

#include <stdbool.h>
#include <time.h>

int myRadix;
bool firstFFT;
bool secondFFT;
bool firstTranspose;
bool secondTranspose;
bool thirdTranspose;
int myRank;
int fft2Radix;
int pNum;
bool dmr;	// whether has a small size FFT between FFT2.1 and FFT2.2

clock_t t1;

R * checksum1;	// checksum for input data
R * checksum2;
R * simpleCoeff;
R * rA;
R * rAcoeff;
R * outChecksum1;	// intermediate checksum after fft1 or fft2 first part
R * outChecksum2;
R * fft2checksum1;	// checksum for fft2.1 input
R * fft2checksum2;
R * outputChecksum1;	// checksum for output before the third tranpose
R * outputChecksum2;	

X(plan) X(plan_dft)(int rank, const int *n,
		    C *in, C *out, int sign, unsigned flags)
{
	pNum = 0;
	fft2Radix = 0;
	myRadix = 0;
	myRank = -1;
	firstFFT = false;
	secondFFT = false;
	firstTranspose = false;
	secondTranspose = false;
	thirdTranspose = false;
	dmr = false;

	checksum1 = NULL;
	checksum2 = NULL;
	simpleCoeff = NULL;
	rA = NULL;
	rAcoeff = NULL;
	outChecksum1 = NULL;
	outChecksum2 = NULL;
	fft2checksum1 = NULL;
	fft2checksum2 = NULL;
	outputChecksum1 = NULL;
	outputChecksum2 = NULL;
	
     return X(plan_many_dft)(rank, n, 1,
			     in, 0, 1, 1, 
			     out, 0, 1, 1, 
			     sign, flags);
}
