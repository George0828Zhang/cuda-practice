#include <cstdio>
#include <cassert>
#define MAXN 2000
#define MULSIDE 16 // each block has size SIDE x SIDE

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

#define get(A, x, y, pitch, N) ((x) >= 0 && (x) < N && (y) >= 0 && (y) < N) ? A[(x)*pitch + (y)] : 0
__global__ void game_of_life_iter(char* A, char* B, size_t pitcha, size_t pitchb, int N){
    int localx = threadIdx.x;
    int localy = threadIdx.y;
    int globalx = localx + blockIdx.x * MULSIDE;
    int globaly = localy + blockIdx.y * MULSIDE;
 
    char alive = 0;
    char self = get(A, globalx, globaly, pitcha, N);
    alive += get(A, globalx-1, globaly-1, pitcha, N);
    alive += get(A, globalx-1, globaly, pitcha, N);
    alive += get(A, globalx-1, globaly+1, pitcha, N);

    alive += get(A, globalx, globaly-1, pitcha, N);
    alive += get(A, globalx, globaly+1, pitcha, N);

    alive += get(A, globalx+1, globaly-1, pitcha, N);
    alive += get(A, globalx+1, globaly, pitcha, N);
    alive += get(A, globalx+1, globaly+1, pitcha, N);
    
    if (self && (alive < 2 || alive > 3)){
        B[globalx * pitchb + globaly] = 0;
    }
    else if (!self && alive == 3){
        B[globalx * pitchb + globaly] = 1;
    }
    else{
        B[globalx * pitchb + globaly] = self;
    }
}


__global__ void faster_game_of_life_iter(char* A, char* B, size_t pitcha, size_t pitchb, int N){
    __shared__ char source[MULSIDE+2][MULSIDE+2];

    int localx = threadIdx.x;
    int localy = threadIdx.y;
    int globalx = localx + blockIdx.x * MULSIDE;
    int globaly = localy + blockIdx.y * MULSIDE;

    // index used to address the 'source' array.
    int src_x = localx + 1;
    int src_y = localy + 1;
    source[src_x][src_y] = get(A, globalx, globaly, pitcha, N);

    // sides
    if (localx == 0)
        source[src_x-1][src_y] = get(A, globalx-1, globaly, pitcha, N);
    if (localx == MULSIDE-1)
        source[src_x+1][src_y] = get(A, globalx+1, globaly, pitcha, N);
    if (localy == 0)
        source[src_x][src_y-1] = get(A, globalx, globaly-1, pitcha, N);
    if (localy == MULSIDE-1)
        source[src_x][src_y+1] = get(A, globalx, globaly+1, pitcha, N);

    // corners
    if (localx == 0 && localy == 0)
        source[src_x-1][src_y-1] = get(A, globalx-1, globaly-1, pitcha, N);
    if (localx == MULSIDE-1 && localy == 0)
        source[src_x+1][src_y-1] = get(A, globalx+1, globaly-1, pitcha, N);
    if (localx == 0 && localy == MULSIDE-1)
        source[src_x-1][src_y+1] = get(A, globalx-1, globaly+1, pitcha, N);
    if (localx == MULSIDE-1 && localy == MULSIDE-1)
        source[src_x+1][src_y+1] = get(A, globalx+1, globaly+1, pitcha, N);
    __syncthreads();
 
    // count alive neighbors
    // unroll to speed up
    char self = source[src_x][src_y];
    char alive = source[src_x-1][src_y-1] + source[src_x-1][src_y] + source[src_x-1][src_y+1]
    + source[src_x][src_y-1] + source[src_x][src_y+1]
    + source[src_x+1][src_y-1] + source[src_x+1][src_y] + source[src_x+1][src_y+1];

    if (self && (alive < 2 || alive > 3)){
        B[globalx * pitchb + globaly] = 0;
    }
    else if (!self && alive == 3){
        B[globalx * pitchb + globaly] = 1;
    }
    else{
        B[globalx * pitchb + globaly] = self;
    }
}
void copyto(char* dst, char* src, size_t pitch){
    gpuErrchk(cudaMemcpy2D((void*)dst, pitch, (void *)src, MAXN, MAXN*sizeof(char), MAXN, cudaMemcpyHostToDevice));
}
void copyback(char* dst, char* src, size_t pitch){
    gpuErrchk(cudaMemcpy2D((void*)dst, MAXN, (void *)src, pitch, MAXN*sizeof(char), MAXN, cudaMemcpyDeviceToHost));
} 
void cuClear(char* dst, size_t pitch){
    gpuErrchk(cudaMemset2D((void*)dst, pitch, 0, MAXN*sizeof(char), MAXN));
}
void print_matrix(int N, char A[]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%d", A[i*MAXN + j]);
        printf("\n");
    }
}
int divCeil(int a, int b){
    int c = a / b;
    if (c * b < a){
        c++;
    }
    return c;
}
char A[MAXN*MAXN];
int main(int argc, char** argv)
{
    char digits[MAXN];
    int N, M, s;
    s = scanf("%d %d", &N, &M);
    for(int i = 0; i < N; i++){
        s = scanf("%s", digits);
        assert(s>0);
        for(int j = 0; j < N; j++){
            A[i*MAXN + j] = digits[j]=='0' ? 0 : 1;
        }
    }

    size_t pitch[2];
    char *devA[2];
    gpuErrchk(cudaMallocPitch(&devA[0], &pitch[0], MAXN*sizeof(char), MAXN));
    gpuErrchk(cudaMallocPitch(&devA[1], &pitch[1], MAXN*sizeof(char), MAXN));
    copyto(devA[0], (char*)A, pitch[0]);

    for (int i = 0; i < M; i++){
        int x = i%2;
        int BLOCKS = divCeil(N, MULSIDE);
        faster_game_of_life_iter <<< dim3(BLOCKS,BLOCKS), dim3(MULSIDE,MULSIDE) >>> (devA[x], devA[!x], pitch[x], pitch[!x], N);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }
    copyback((char*)A, devA[M%2], pitch[M%2]);
    print_matrix(N, A);


    return 0;	
}
