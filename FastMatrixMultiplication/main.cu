#include <stdio.h>
#include <stdint.h>
// #define DEBUG
#define UINT uint32_t
#define MAXN 1024

#define MULSIDE 16 // each block has size SIDE x SIDE
#define MULBLK (MAXN / MULSIDE)  // divide C into BLK x BLK blocks

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


 
__global__ void mul_kernel(UINT* A, UINT* B, UINT* C, size_t pitcha, size_t pitchb, size_t pitchc){
    __shared__ UINT left[MULSIDE][MULSIDE];
    __shared__ UINT right[MULSIDE][MULSIDE];
 
    int gridx = blockIdx.x;
    int gridy = blockIdx.y;
 
    int localx = threadIdx.x;
    int localy = threadIdx.y;
 
    int globalx = gridx * MULSIDE + localx;//x for C
    int globaly = gridy * MULSIDE + localy;//y for C
 
 
    UINT result = 0;
    for(int block = 0; block < MULBLK; block++){
        // recommended way to address cuda matrix is to use pitch, syntax below:
        // T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;
        // also, when loading, transpose the right matrix for temporal locality
        left[localx][localy] = *((UINT*)((char*)A + globalx * pitcha) + (block*MULSIDE + localy));
        right[localy][localx] = *((UINT*)((char*)B + (block*MULSIDE + localx) * pitchb) + globaly);
        __syncthreads();
 
        for(int k = 0; k < MULSIDE; k++){
            result += left[localx][k] * right[localy][k];
        }
        __syncthreads();
    }  
 
    *((UINT*)((char*)C + globalx * pitchc) + globaly) = result;
}
void copyto(UINT* dst, UINT* src, size_t pitch){
    gpuErrchk(cudaMemcpy2D((void*)dst, pitch, (void *)src, pitch, MAXN*sizeof(UINT), MAXN, cudaMemcpyHostToDevice));
}
void copyback(UINT* dst, UINT* src, size_t pitch){
    gpuErrchk(cudaMemcpy2D((void*)dst, pitch, (void *)src, pitch, MAXN*sizeof(UINT), MAXN, cudaMemcpyDeviceToHost));
} 
void cuClear(UINT* dst, size_t pitch){
    gpuErrchk(cudaMemset2D((void*)dst, pitch, 0, MAXN*sizeof(UINT), MAXN));
}
void clear(UINT A[][MAXN]){
    for (int i = 0; i < MAXN; i++) {
        memset(A, 0, MAXN);
    }
}

void rand_gen(UINT c, int N, UINT A[][MAXN]) {
    UINT x = 2, n = N*N;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            x = (x * x + c + i + j)%n;
            A[i][j] = x;
        }
    }
}
void print_matrix(int N, UINT A[][MAXN]) {
    for (int i = 0; i < N; i++) {
        fprintf(stderr, "[");
        for (int j = 0; j < N; j++)
            fprintf(stderr, " %u", A[i][j]);
        fprintf(stderr, " ]\n");
    }
}
UINT signature(int N, UINT A[][MAXN]) {
    UINT h = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            h = (h + A[i][j]) * 2654435761LU;
    }
    return h;
}
UINT A[MAXN][MAXN], B[MAXN][MAXN], C[MAXN][MAXN];
int main() {
    int N;
    uint32_t S1, S2;
    scanf("%d %u %u", &N, &S1, &S2);
    rand_gen(S1, N, A);
    rand_gen(S2, N, B);

    size_t pitcha, pitchb, pitchc;
    UINT *devA, *devB, *devC;
    gpuErrchk(cudaMallocPitch(&devA, &pitcha, MAXN*sizeof(UINT), MAXN));
    gpuErrchk(cudaMallocPitch(&devB, &pitchb, MAXN*sizeof(UINT), MAXN));
    gpuErrchk(cudaMallocPitch(&devC, &pitchc, MAXN*sizeof(UINT), MAXN));

    copyto(devA, (UINT*)A, pitcha);
    copyto(devB, (UINT*)B, pitchb);
    cuClear(devC, pitchc);

    mul_kernel <<< dim3(MULBLK,MULBLK), dim3(MULSIDE,MULSIDE) >>> (devA, devB, devC, pitcha, pitchb, pitchc);//AB
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    copyback((UINT*)C, devC, pitchc);


#ifdef DEBUG
    print_matrix(N, A);
    print_matrix(N, B);
    print_matrix(N, C);
#endif
    printf("%u\n", signature(N, C));
    return 0;
}