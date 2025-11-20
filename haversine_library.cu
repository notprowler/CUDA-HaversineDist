#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

#define R_earth 6371.0 

//use any references to compute haversine distance bewtween (x1,y1) and (x2,y2), given in vectors/arrays
__global__ void haversine_distance_kernel(int size, 
                                          const double *x1,
                                          const double *y1, 
                                          const double *x2,
                                          const double *y2, 
                                          double *dist) 
  {
// Hav(0) = hav(DeltaLat) + cos(lat1) * cos(lat2) * hav(DeltaLong)
// Hav(0) = sin^2(0/2)
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;

    double lat1 = y1[idx] * M_PI / 180.0;
    double lon1 = x1[idx] * M_PI / 180.0;
    double lat2 = y2[idx] * M_PI / 180.0;
    double lon2 = x2[idx] * M_PI / 180.0;

    double dlat = lat2 - lat1;
    double dlon = lon2 - lon1;

    // a = sin²(dlat/2) + cos(lat1)*cos(lat2)*sin²(dlon/2)

    double s1 = sin(dlat * 0.5);
    double s2 = sin(dlon * 0.5);

    double a = s1*s1 + cos(lat1) * cos(lat2) * s2*s2;
      
    // c = 2 * atan2(sqrt(a), sqrt(1 - a))
    double c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a));

    dist[idx] = R_earth * c;
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
