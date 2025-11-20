#include <sstream>
#include <iostream>
#include <cuda_runtime.h>

__global__ void haversine_distance_kernel(int size, const double *x1,const double *y1,
    const double *x2,const double *y2, double *dist)
{
 //use any references to compute haversine distance bewtween (x1,y1) and (x2,y2), given in vectors/arrays
 //e.g., https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula
}


void run_kernel(int size, const double *x1,const double *y1, const double *x2,const double *y2, double *dist)
   
{
  dim3 dimBlock(1024);
  printf("in run_kernel dimBlock.x=%d\n",dimBlock.x);

  dim3 dimGrid(ceil((double)size / dimBlock.x));
  
  haversine_distance_kernel<<<dimGrid, dimBlock>>>
    (size,x1,y1,x2,y2,dist);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::stringstream strstr;
    strstr << "run_kernel launch failed" << std::endl;
    strstr << "dimBlock: " << dimBlock.x << ", " << dimBlock.y << std::endl;
    strstr << "dimGrid: " << dimGrid.x << ", " << dimGrid.y << std::endl;
    strstr << cudaGetErrorString(error);
    throw strstr.str();
  }
}
