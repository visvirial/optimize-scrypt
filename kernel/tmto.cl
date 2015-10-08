// vim: set filetype=c:
/**
 * Optimizations: loop-unrolling, concurrent-memory-access, tmto
 */

#include "ocl_common.h"

#define TMTO_FACTOR (2)
#if SCRYPT_N % TMTO_FACTOR != 0
#  error "TMTO_FACTOR should divide SCRYPT_N!"
#endif

// Returns the address of n-th element on scratchpad.
#define addr(n, i) (n*(8*GLOBAL_WORK_SIZE) + 8*offset + i)

void scrypt_core(uint4 X[8], __global void *scratchpad) {
	const size_t offset = get_global_id(0) - get_global_offset(0);
	__global uint4 *V = (__global uint4*)scratchpad;
	for(size_t i=0; i<SCRYPT_N; i+=TMTO_FACTOR) {
		// V_i = X
		#pragma unroll 8
		for(size_t k=0; k<8; k++) {
			V[addr(i/TMTO_FACTOR, k)] = X[k];
		}
		// X = H(X)
		#pragma unroll
		for(size_t k=0; k<TMTO_FACTOR; k++) {
			salsa(X);
		}
	}
	for(size_t i=0; i<SCRYPT_N; i++) {
		// j = Integerify(X) % N
		size_t j = X[7].x & (SCRYPT_N - 1);
		// X = H(X xor V_j)
		uint4 VV[8];
		#pragma unroll 8
		for(size_t k=0; k<8; k++) {
			VV[k] = V[addr(j/TMTO_FACTOR, k)];
		}
		#pragma unroll
		for(size_t k=1; k<TMTO_FACTOR; k++) {
			if(j%TMTO_FACTOR >= k) salsa(VV);
		}
		#pragma unroll 8
		for(size_t k=0; k<8; k++) {
			X[k] ^= VV[k];
		}
		salsa(X);
	}
}

