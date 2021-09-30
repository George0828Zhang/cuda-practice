#include <cstdio>

int main(void)
{
  int count;
  cudaGetDeviceCount(&count);

  printf("%d devices found supporting CUDA\n", count);

  char split[] = "----------------------------------\n";

  cudaDeviceProp p;
  for(int d = 0; d < count; d++){
    cudaGetDeviceProperties(&p, d);
    printf("%s", split);
    printf("Device %s\n", p.name);
    printf("%s", split);
    printf(" Device memory: \t%zu\n", p.totalGlobalMem);
    printf(" Memory per-block: \t%lu\n", p.sharedMemPerBlock);
    printf(" Register per-block: \t%d\n", p.regsPerBlock);
    printf(" Warp size: \t\t%d\n", p.warpSize);
    printf(" Memory pitch: \t\t%lu\n", p.memPitch);
    printf(" Constant Memory: \t%lu\n", p.totalConstMem);
    printf(" Max thread per-block: \t%d\n", p.maxThreadsPerBlock);
    printf(" Max thread dim: \t%d / %d / %d\n", p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
    printf(" Max grid size: \t%d / %d / %d\n", p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
    printf(" Ver: \t\t\t%d.%d\n", p.major, p.minor);
    printf(" Clock: \t\t%d\n", p.clockRate);
    printf(" Texture Alignment: \t%lu\n", p.textureAlignment);
  }
  
  return 0;	
}
