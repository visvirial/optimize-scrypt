// vim: set filetype=c:

#include "ocl_common.h"

#define CONCURRENT_THREADS (8192)
#define LOOKUP_GAP (2)

#define Coord(x,y,z) x+y*(x ## SIZE)+z*(y ## SIZE)*(x ## SIZE)
#define CO Coord(z,x,y)

void scrypt_core(uint4 X[8], __global void *scratchpad) {
	__global uint4 *lookup = scratchpad;
	const uint zSIZE = 8;
	//const uint ySIZE = (SCRYPT_N/LOOKUP_GAP+(SCRYPT_N%LOOKUP_GAP>0));
	const uint xSIZE = CONCURRENT_THREADS;
	uint x = get_global_id(0)%xSIZE;
	
	for(uint y=0; y<SCRYPT_N/LOOKUP_GAP; ++y) {
#pragma unroll
		for(uint z=0; z<zSIZE; ++z)
			lookup[CO] = X[z];
		for(uint i=0; i<LOOKUP_GAP; ++i)
			salsa(X);
	}
#if (LOOKUP_GAP != 1) && (LOOKUP_GAP != 2) && (LOOKUP_GAP != 4) && (LOOKUP_GAP != 8)
	{
		uint y = (SCRYPT_N/LOOKUP_GAP);
#pragma unroll
		for(uint z=0; z<zSIZE; ++z)
			lookup[CO] = X[z];
		for(uint i=0; i<SCRYPT_N%LOOKUP_GAP; ++i)
			salsa(X);
	}
#endif
	for (uint i=0; i<SCRYPT_N; ++i) {
		uint4 V[8];
		uint j = X[7].x & (SCRYPT_N-1);
		uint y = (j/LOOKUP_GAP);
#pragma unroll
		for(uint z=0; z<zSIZE; ++z)
			V[z] = lookup[CO];
#if (LOOKUP_GAP == 1)
#elif (LOOKUP_GAP == 2)
		if (j&1)
			salsa(V);
#else
		uint val = j%LOOKUP_GAP;
		for (uint z=0; z<val; ++z)
			salsa(V);
#endif
#pragma unroll
		for(uint z=0; z<zSIZE; ++z)
			X[z] ^= V[z];
		salsa(X);
	}
}

