// vim: set filetype=c:
/**
 * Optimizations: loop-unrolling
 */

#include "ocl_common.h"

void scrypt_core(uint4 X[8], __global void *scratchpad) {
	for(size_t i=0; i<2*SCRYPT_N; i++) {
		salsa(X);
	}
}

