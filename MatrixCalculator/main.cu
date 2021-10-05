#include <stdio.h>
#include <stdint.h>
// #define DEBUG
#define UINT uint32_t
#define MAXN 1024
 
#define MULSIDE 16 // each block has size SIDE x SIDE
#define MULBLK (MAXN / MULSIDE)  // divide C into BLK x BLK blocks
 
#define ADDSIDE 256
#define ADDBLK (MAXN*(MAXN / ADDSIDE))
 
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
 
__device__ static inline UINT get(int N, UINT* M, int x, int y){
    return (x < N && y < N) ? M[x*N + y] : 0;
}
__device__ static inline UINT set(int N, UINT* M, int x, int y, UINT v){
    if (x < N && y < N)
        M[x*N + y] = v;
    return;
}
__global__ void mul_kernel(int N, UINT* A, UINT* B, UINT* C){
    __shared__ UINT left[MULSIDE][MULSIDE];
    __shared__ UINT right[MULSIDE][MULSIDE];
 
    int localx = threadIdx.x;
    int localy = threadIdx.y;
 
    int globalx = blockIdx.x * MULSIDE + localx;//x for C
    int globaly = blockIdx.y * MULSIDE + localy;//y for C
 
    UINT result = 0;
    for(int block = 0; block < MULBLK; block++){
        left[localx][localy] = get(N, A, globalx, (block*MULSIDE + localy));
        right[localy][localx] = get(N, B, block*MULSIDE + localx, globaly);
        __syncthreads();
 
        for(int k = 0; k < MULSIDE; k++){
            result += left[localx][k] * right[localy][k];
        }
        __syncthreads();
    }
 
    set(N, C, globalx, globaly, result);
}
__global__ void add_kernel(int N, UINT* A, UINT* B, UINT* C){
    int index = blockIdx.x * ADDSIDE + threadIdx.x;
    if (index < N)
        C[index] = A[index] + B[index];
}
 
void rand_gen(UINT c, int N, UINT* A) {
    UINT x = 2, n = N*N;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            x = (x * x + c + i + j)%n;
            A[i*N + j] = x;
        }
    }
}
void print_matrix(int N, UINT* A) {
    for (int i = 0; i < N; i++) {
        fprintf(stderr, "[");
        for (int j = 0; j < N; j++)
            fprintf(stderr, " %u", A[i*N + j]);
        fprintf(stderr, " ]\n");
    }
}
UINT signature(int N, UINT* A) {
    UINT h = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            h = (h + A[i*N + j]) * 2654435761LU;
    }
    return h;
}
UINT hostIN[6][MAXN*MAXN], hostTMP[2][MAXN*MAXN];
int main() {
    int N, S[6];
    scanf("%d", &N);
    for (int i = 0; i < 6; i++) {
        scanf("%d", &S[i]);
        rand_gen(S[i], N, hostIN[i]);
    }
 
    UINT *IN[6], *TMP[6];
    size_t matsz = sizeof(UINT) * N * N;
    for (int i = 0; i < 6; i++){
        gpuErrchk(cudaMalloc(&IN[i], matsz));
        gpuErrchk(cudaMemcpy(IN[i], hostIN[i], matsz, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMalloc(&TMP[i], matsz));
    }
 
    // AB
    mul_kernel <<< dim3(MULBLK,MULBLK), dim3(MULSIDE,MULSIDE) >>> (N, IN[0], IN[1], TMP[0]);
    // CD
    mul_kernel <<< dim3(MULBLK,MULBLK), dim3(MULSIDE,MULSIDE) >>> (N, IN[2], IN[3], TMP[1]);
    // AB+CD
    add_kernel <<< ADDBLK, ADDSIDE >>> (N*N, TMP[0], TMP[1], TMP[2]);
    // ABE
    mul_kernel <<< dim3(MULBLK,MULBLK), dim3(MULSIDE,MULSIDE) >>> (N, TMP[0], IN[4], TMP[3]);
    // CDF
    mul_kernel <<< dim3(MULBLK,MULBLK), dim3(MULSIDE,MULSIDE) >>> (N, TMP[1], IN[5], TMP[4]);
    // ABE+CDF
    add_kernel <<< ADDBLK, ADDSIDE >>> (N*N, TMP[3], TMP[4], TMP[5]);
 
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
 
    gpuErrchk(cudaMemcpy(hostTMP[0], TMP[2], matsz, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(hostTMP[1], TMP[5], matsz, cudaMemcpyDeviceToHost));
 
    printf("%u\n", signature(N, hostTMP[0]));
    printf("%u\n", signature(N, hostTMP[1]));
 
    return 0;
}