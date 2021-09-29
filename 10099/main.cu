#include <stdio.h>
#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
// #include "utils.h"
 
#define CHUNKSIZE 16
#define THREADS_PER_BLOCK 256
#define MAXN 16777216
#define MAXBLOCKS (MAXN / CHUNKSIZE / THREADS_PER_BLOCK)
uint32_t A[MAXN], B[MAXN], C[MAXN];
 
// function for debugging.
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
 
__device__ uint32_t rotate_left(uint32_t x, uint32_t n) {
    return  (x << n) | (x >> (32-n));
}
__device__ uint32_t encrypt(uint32_t m, uint32_t key) {
    return (rotate_left(m, key&31) + key)^key;
}
 
// 1. kernel
__global__ void mul(int N, uint32_t key1, uint32_t key2, uint32_t* out){
    int chunk_id = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t res = 0;
    // process chunk
    int start = chunk_id * CHUNKSIZE;
    int end = (chunk_id+1)*CHUNKSIZE;
    end = N < end ? N : end;
    for(int k = start; k < end; k++){
        res += encrypt(k, key1) * encrypt(k, key2);
    }
    out[chunk_id] = res;
}

// 2. add reduction
__global__ void mul_reduce(int N, uint32_t key1, uint32_t key2, uint32_t* out){
    __shared__ uint32_t sdata[THREADS_PER_BLOCK]; // for reduction

    int tid = threadIdx.x;
    int chunk_id = threadIdx.x + blockIdx.x * blockDim.x;
    // process chunk
    int start = chunk_id * CHUNKSIZE;
    int end = (chunk_id+1)*CHUNKSIZE;
    end = N < end ? N : end;
    sdata[tid] = 0;
    for(int k = start; k < end; k++){
        sdata[tid] += encrypt(k, key1) * encrypt(k, key2);
    }
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}
 
int divCeil(int a, int b){
    int c = a / b;
    if (c * b < a){
        c++;
    }
    return c;
}
 
int main(int argc, char *argv[]) {
    int N, N_CHUNKS, BLOCKS;
    uint32_t key1, key2;
 
    uint32_t *devC;
 
    gpuErrchk(cudaMalloc(&devC, sizeof(uint32_t) * MAXN));
 
    while (scanf("%d %" PRIu32 " %" PRIu32, &N, &key1, &key2) == 3) {
        N_CHUNKS = divCeil(N, CHUNKSIZE);
        BLOCKS = divCeil(N_CHUNKS, THREADS_PER_BLOCK);
        mul_reduce <<< BLOCKS, THREADS_PER_BLOCK >>> (N, key1, key2, devC);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
 
        // copyback and sum
        gpuErrchk(cudaMemcpy(C, devC, sizeof(uint32_t) * BLOCKS, cudaMemcpyDeviceToHost));
 
        uint32_t sum = 0;
        for (int i = 0; i < BLOCKS; i++){
            sum += C[i];
        }
        printf("%" PRIu32 "\n", sum);
    }
 
    cudaFree(devC);
    return 0;
}