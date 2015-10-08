// vim: set filetype=c:
/**
 * Optimizations: loop-unrolling
 */

#include "ocl_common.h"

void scrypt_core(uint4 X[8], __global void *scratchpad) {
	const size_t offset = get_global_id(0) - get_global_offset(0);
	__global uint4 *V = &((__global uint4*)scratchpad)[offset * (8 * SCRYPT_N)];
	#pragma unroll 32
	for(size_t i=0; i<SCRYPT_N; i++) {
		// V_i = X
		#pragma unroll 8
		for(size_t k=0; k<8; k++) {
			V[8*i+k] = X[k];
		}
		// X = H(X)
		salsa(X);
	}
	#pragma unroll 32
	for(size_t i=0; i<SCRYPT_N; i++) {
		size_t j;
		// j = Integerify(X) % N
		j = X[7].x & (SCRYPT_N - 1);
		// X = H(X xor V_j)
		#pragma unroll 8
		for(size_t k=0; k<8; k++) {
			X[k] ^= V[8*j+k];
		}
		salsa(X);
	}
}

