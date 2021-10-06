#include <stdio.h>
#include <stdint.h>
#include <assert.h>
// #define DEBUG
#define UINT uint32_t
#define TOPM 26
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
__device__ static inline void set(int N, UINT* M, int x, int y, UINT v){
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
UINT hostIN[TOPM][MAXN*MAXN], hostTMP[MAXN*MAXN];
int main() {
    int M, N, S[TOPM], s;
    s = scanf("%d %d", &M, &N);
    for (int i = 0; i < M; i++) {
        s = scanf("%d", &S[i]);
        rand_gen(S[i], N, hostIN[i]);
    }
 
    UINT *IN[TOPM];
    UINT *TMP[TOPM];
    size_t matsz = sizeof(UINT) * N * N;
    for (int i = 0; i < M; i++){
        gpuErrchk(cudaMalloc(&IN[i], matsz));
        gpuErrchk(cudaMemcpy(IN[i], hostIN[i], matsz, cudaMemcpyHostToDevice));
    }
    for (int i = 0; i < TOPM; i++)
        gpuErrchk(cudaMalloc(&TMP[i], matsz));
 
    int Q;
    char E[26];
    s = scanf("%d", &Q);
    for (int i = 0; i < Q; i++) {
        s = scanf("%s", E);
        UINT *addbuf = NULL, *mulbuf = NULL, *r_operand;
        for (int j = 0, tmp_id = 0; j < 27; j++){
            // fprintf(stderr, "j=%d, \'%c\', tmp=%d\n", j, E[j]=='\0' ? '#' : E[j], tmp_id);
            if(j < 26 && E[j] != '\0' && E[j] != '+'){
                // before +
                if (mulbuf == NULL){
                    // first operand
                    mulbuf = IN[E[j]-'A'];
                }
                else{
                    // right operand
                    r_operand = IN[E[j]-'A'];
                    mul_kernel <<< dim3(MULBLK,MULBLK), dim3(MULSIDE,MULSIDE) >>> 
                        (N, mulbuf, r_operand, TMP[tmp_id]);
                    mulbuf = TMP[tmp_id++];
                }
            }
            else if(addbuf == NULL){
                // first segment e.g. "ABCD"+EF...
                addbuf = mulbuf;
                mulbuf = NULL;
            }
            else{
                // new segment e.g. ABCD+"EF"...
                add_kernel <<< ADDBLK, ADDSIDE >>> (N*N, addbuf, mulbuf, TMP[tmp_id]);
                addbuf = TMP[tmp_id++];
                mulbuf = NULL;
            }
 
            if (j >= 26 || E[j] == '\0')
                break;
 
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
        }
        gpuErrchk(cudaMemcpy(hostTMP, addbuf, matsz, cudaMemcpyDeviceToHost));
        printf("%u\n", signature(N, hostTMP));
    }
    assert(s>0);
 
    for (int i = 0; i < M; i++){
        gpuErrchk(cudaFree(IN[i]));
    }
    for (int i = 0; i < TOPM; i++){
        gpuErrchk(cudaFree(TMP[i]));
    }
    return 0;
}