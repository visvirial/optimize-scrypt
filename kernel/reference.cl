// vim: set filetype=c:
/**
 * Optimizations: none.
 */

#include "ocl_common.h"

void scrypt_core(uint4 X[8], __global void *scratchpad) {
	const size_t offset = get_global_id(0) - get_global_offset(0);
	__global uint4 *V = &((__global uint4*)scratchpad)[offset * (8 * SCRYPT_N)];
	for(size_t i=0; i<SCRYPT_N; i++) {
		// V_i = X
		for(size_t k=0; k<8; k++) {
			V[8*i+k] = X[k];
		}
		// X = H(X)
		salsa(X);
	}
	for(size_t i=0; i<SCRYPT_N; i++) {
		// j = Integerify(X) % N
		size_t j = X[7].x & (SCRYPT_N - 1);
		// X = H(X xor V_j)
		for(size_t k=0; k<8; k++) {
			X[k] ^= V[8*j+k];
		}
		salsa(X);
	}
}

