// vim: set filetype=c:
/**
 * Optimizations: loop-unrolling, concurrent-memory-access
 */

#include "ocl_common.h"

// Returns the address of n-th element on scratchpad.
//#define addr(n, i) (offset*(8*SCRYPT_N) + 8*n + i)
#define addr(n, i) (n*(8*GLOBAL_WORK_SIZE) + 8*offset + i)

void scrypt_core(uint4 X[8], __global void *scratchpad) {
	const size_t offset = get_global_id(0) - get_global_offset(0);
	__global uint4 *V = (__global uint4*)scratchpad;
	for(size_t i=0; i<SCRYPT_N; i++) {
		// V_i = X
		#pragma unroll 8
		for(size_t k=0; k<8; k++) {
			V[addr(i, k)] = X[k];
		}
		// X = H(X)
		salsa(X);
	}
	for(size_t i=0; i<SCRYPT_N; i++) {
		// j = Integerify(X) % N
		size_t j = X[7].x & (SCRYPT_N - 1);
		// X = H(X xor V_j)
		#pragma unroll 8
		for(size_t k=0; k<8; k++) {
			X[k] ^= V[addr(j, k)];
		}
		salsa(X);
	}
}

