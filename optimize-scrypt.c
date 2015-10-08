
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <signal.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

#include <CL/cl.h>
#include <libscrypt.h>

#include "common.h"

#define THREADS (2)

static char *kernel_src;
static size_t kernel_size;

static cl_context context;
static cl_program program;
static cl_command_queue command_queue[THREADS];
static cl_kernel kernel[THREADS];
static cl_mem cl_mem_output[THREADS];
static cl_mem cl_mem_scratchpad[THREADS];

static uint8_t answers[GLOBAL_WORK_SIZE][32];

static uint64_t hashes[THREADS];

static uint64_t microtime() {
	struct timeval time;
	gettimeofday(&time, NULL);
	return (uint64_t)time.tv_sec * 1000 * 1000 + time.tv_usec;
}

static void scrypt_cpu(const uint8_t *in, uint8_t *out) {
	libscrypt_scrypt(
		// passwd, passwdlen
		in, INPUT_LEN,
		// salt, saltlen
		in, INPUT_LEN,
		// N, r, p
		SCRYPT_N, SCRYPT_r, SCRYPT_p,
		// buf, buflen
		out, 32
	);
}

static void print_hash(const uint8_t *hash) {
	printf("0x");
	for(size_t i=0; i<32; i++) {
		printf("%02x", hash[31-i]);
	}
	printf("\n");
}

static void ocl_check_err(cl_int errno, char *msg) {
	if(errno == CL_SUCCESS)
		return;
	fprintf(stderr, "E: %s.\n", msg);
	char *reason = "unknown reason";
	switch(errno){
		case CL_INVALID_COMMAND_QUEUE:
			reason = "command queue is not valid";
			break;
		case CL_INVALID_CONTEXT:
			reason = "context is not valid";
			break;
		case CL_INVALID_MEM_OBJECT:
			reason = "memory object is not valid";
			break;
		case CL_INVALID_VALUE:
			reason = "some value is not valid";
			break;
		case CL_INVALID_EVENT_WAIT_LIST:
			reason = "event wait list is not valid";
			break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:
			reason = "failed to allocate memory object";
			break;
		case CL_OUT_OF_RESOURCES:
			reason = "failed to allocate resources on the device";
			break;
		case CL_OUT_OF_HOST_MEMORY:
			reason = "failed to allocate memory on the host";
			break;
	}
	fprintf(stderr, "E: %s (%d).\n", reason, errno);
	exit(1);
}

#pragma GCC diagnostic ignored "-Wunused-parameter"
static void ocl_pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data) {
	fprintf(stderr, "W: caught an error in ocl_pfn_notify:\nW:   %s", errinfo);
}
#pragma GCC diagnostic pop

void ocl_deinit() {
	//clFlush(command_queue);
	clReleaseContext(context);
	clReleaseProgram(program);
	for(size_t i=0; i<THREADS; i++) {
		clReleaseCommandQueue(command_queue[i]);
		clReleaseKernel(kernel[i]);
	}
}

static bool terminate = false;

static void signal_handler() {
	printf("W: Ctrl-C detected. terminating...\n");
	terminate = true;
}

void* run_thread(void *params) {
	const size_t id = *(size_t*)params;
	//
	// Prepare.
	printf("I[%ld] thread started.\n", id);
	//
	// Run.
	const size_t nonce_offset = 0;
	// Run kernel.
	hashes[id] = 0;
	uint64_t t = microtime();
	uint64_t hash = 0;
	for(; !terminate;) {
		// Report current hashrate.
		if(microtime() - t > 3 * 1000 * 1000) {
			printf("I[%ld]: current hashrate = %.0fkH/s (3s avg.)\n", id, 1e3 * hash / (microtime() - t));
			t = microtime();
			hash = 0;
		}
		cl_event event;
		const size_t gws = GLOBAL_WORK_SIZE;
		const size_t lws = LOCAL_WORK_SIZE;
		cl_int errno;
		errno = clEnqueueNDRangeKernel(command_queue[id], kernel[id], 1, &nonce_offset, &gws, &lws, 0, NULL, &event);
		ocl_check_err(errno, "failed to execute clEnqueueNDRangeKernel");
		// Checks for answer.
		// Get result.
		uint8_t output[32*GLOBAL_WORK_SIZE];
		errno = clEnqueueReadBuffer(command_queue[id], cl_mem_output[id], CL_TRUE, 0, 32*GLOBAL_WORK_SIZE, output, 1, &event, NULL);
		ocl_check_err(errno, "failed to execute clEnqueueReadBuffer");
		// Check for validity.
		for(size_t i=0; i<GLOBAL_WORK_SIZE; i++) {
			const uint32_t nonce = nonce_offset + i;
			uint8_t *result = output + 32 * i;
			if(strncmp((char*)answers[i], (char*)result, 32) != 0) {
				printf("I[%ld]: hash mismatch found for nonce=0x%08x (index=%zd)!\n", id, nonce, i);
				printf("Answer: ");
				print_hash(answers[i]);
				printf("Result: ");
				print_hash(result);
				break;
			}
		}
		// Increment hash counter.
		hashes[id] += GLOBAL_WORK_SIZE;
		hash += GLOBAL_WORK_SIZE;
	}
	return NULL;
}

int main(int argc, char *argv[]) {
	
	if(argc < 2) {
		printf("usage: %s KERNEL\n", argv[0]);
		return 1;
	}
	
	char *kernel_name = argv[1];
	
	// Load kernel source
	printf("I: initializing OpenCL environment (kernel=%s)...\n", kernel_name);
	#define KERNEL_SOURCE_BUF_SIZE (0x100000)
	char kernel_path[1024];
	sprintf(kernel_path, "kernel/%s.cl", kernel_name);
	FILE *fp = fopen(kernel_path, "r");
	if(!fp){
		fprintf(stderr, "E: failed to load kernel: %s.\n", kernel_name);
		exit(1);
	}
	kernel_src = malloc(KERNEL_SOURCE_BUF_SIZE * sizeof(char));
	kernel_size = fread(kernel_src, 1, KERNEL_SOURCE_BUF_SIZE, fp);
	fclose(fp);
	// Initialize OpenCL.
	cl_int errno;
	// Get platform/device information
	cl_platform_id platforms;
	errno = clGetPlatformIDs(1, &platforms, NULL);
	ocl_check_err(errno, "failed to execute clGetPlatformIDs");
	cl_device_id devices;
	errno = clGetDeviceIDs(platforms, CL_DEVICE_TYPE_GPU, 1, &devices, NULL);
	ocl_check_err(errno, "failed to execute clGetDeviceIDs");
	// Create OpenCL context
	context = clCreateContext(NULL, 1, &devices, ocl_pfn_notify, NULL, &errno);
	ocl_check_err(errno, "failed to execute clCreateContext");
	// Create kernel program from source
	program = clCreateProgramWithSource(context, 1, (const char **)&kernel_src, (const size_t *)&kernel_size, &errno);
	ocl_check_err(errno, "failed to execute clCreateProgramWithSource");
	// Build kernel program
	errno = clBuildProgram(program, 0, NULL, "-w -I.", NULL, NULL);
	{
		char *build_log = malloc(0xFFFF*sizeof(char));;
		size_t log_size;
		clGetProgramBuildInfo(program, devices, CL_PROGRAM_BUILD_LOG, 0xFFFF*sizeof(char), build_log, &log_size);
		printf("I: kernel build log:\n%s\n", build_log);
		free(build_log);
	}
	ocl_check_err(errno, "failed to execute clBuildProgram");
	for(size_t i=0; i<THREADS; i++) {
		// Create command queue
		command_queue[i] = clCreateCommandQueue(context, devices, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &errno);
		ocl_check_err(errno, "failed to execute clCreateCommandQueueWithProperties");
		// Create kernel
		kernel[i] = clCreateKernel(program, "run", &errno);
		ocl_check_err(errno, "failed to execute clCreateKernel");
		// Set argument buffers.
		cl_mem_output    [i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY|CL_MEM_HOST_READ_ONLY, 32*GLOBAL_WORK_SIZE, NULL, NULL);
		cl_mem_scratchpad[i] = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_HOST_NO_ACCESS, GLOBAL_WORK_SIZE*128*SCRYPT_N, NULL, NULL);
		clSetKernelArg(kernel[i], 0, sizeof(cl_mem), &cl_mem_output[i]);
		clSetKernelArg(kernel[i], 1, sizeof(cl_mem), &cl_mem_scratchpad[i]);
	}
	
	// Compute answers.
	printf("I: computing answers...\n");
	for(uint32_t nonce=0; nonce<GLOBAL_WORK_SIZE; nonce++) {
		uint8_t in[INPUT_LEN];
		memcpy(in, test_data, INPUT_LEN);
		memcpy(in+76, &nonce, 4);
		scrypt_cpu(in, answers[nonce]);
	}
	
	if(signal(SIGINT, signal_handler) == SIG_ERR) {
		printf("E: failed to register signal handler.\n");
		exit(1);
	}
	
	uint64_t begin = microtime();
	
	// Launch threads.
	pthread_t threads[THREADS];
	size_t ids[THREADS];
	for(size_t i=0; i<THREADS; i++) {
		ids[i] = i;
		if(pthread_create(&threads[i], NULL, run_thread, &ids[i]) != 0) {
			printf("E: failed to create thread %ld!\n", i);
			exit(1);
		}
	}
	
	// Wait for thread to terminate.
	for(size_t i=0; i<THREADS; i++) {
		pthread_join(threads[i], NULL);
	}
	
	uint64_t total_time = microtime() - begin;
	
	// Print stat.
	uint64_t hashes_calculated = 0;
	for(size_t i=0; i<THREADS; i++) {
		hashes_calculated += hashes[i];
	}
	printf("Hashrate: %.0fkH/s\n", 1e3*hashes_calculated/total_time);
	
	printf("I: ended.\n");
	
	return 0;
}

