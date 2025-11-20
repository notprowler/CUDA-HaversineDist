#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <sys/time.h>

void run_kernel(int size, const double *x1,const double *y1, const double *x2,const double *y2, double *dist);

float calc_time(char *msg,timeval t0, timeval t1)
{
 	long d = t1.tv_sec*1000000+t1.tv_usec - t0.tv_sec * 1000000-t0.tv_usec;
 	float t=(float)d/1000;
 	if(msg!=NULL)
 		printf("%s ...%10.3f\n",msg,t);
 	return t;
}

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void haversine_distance(int size,pybind11::array_t<double> x1_v,pybind11::array_t<double> y1_v,
    pybind11::array_t<double> x2_v,pybind11::array_t<double> y2_v,pybind11::array_t<double> dist_v)
{
  assert(x1_v.request().ndim==1);
  assert(x2_v.request().ndim==1);
  assert(y1_v.request().ndim==1);
  assert(y2_v.request().ndim==1);
  assert(dist_v.request().ndim==1);
  
  timeval t0, t1;

  double *d_x1,*d_y1,*d_x2,*d_y2,*d_dist;
  gettimeofday(&t0, NULL);

  HANDLE_ERROR( cudaMalloc(&d_x1, size * sizeof(double)) );
  HANDLE_ERROR( cudaMalloc(&d_y1, size * sizeof(double)) );
  HANDLE_ERROR( cudaMalloc(&d_x2, size * sizeof(double)) );
  HANDLE_ERROR( cudaMalloc(&d_y2, size * sizeof(double)) );
  HANDLE_ERROR( cudaMalloc(&d_dist, size * sizeof(double)) );
  gettimeofday(&t1, NULL);
  calc_time((char *)"GPU memory allocation time (ms)", t0, t1);

  double* h_x1 = reinterpret_cast<double*>(x1_v.request().ptr);
  double* h_y1 = reinterpret_cast<double*>(y1_v.request().ptr);
  double* h_x2 = reinterpret_cast<double*>(x2_v.request().ptr);
  double* h_y2 = reinterpret_cast<double*>(y2_v.request().ptr);
  double* h_dist = reinterpret_cast<double*>(dist_v.request().ptr);
  
  gettimeofday(&t0, NULL);
  HANDLE_ERROR( cudaMemcpy(d_x1, h_x1, size * sizeof(double), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(d_y1, h_y1, size * sizeof(double), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(d_x2, h_x2, size * sizeof(double), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(d_y2, h_y2, size * sizeof(double), cudaMemcpyHostToDevice) );
  gettimeofday(&t1, NULL);
  calc_time((char *)"CPU → GPU transfer time (ms)", t0, t1);

  gettimeofday(&t0, NULL);
  run_kernel(size, d_x1, d_y1, d_x2, d_y2, d_dist);
  HANDLE_ERROR(cudaDeviceSynchronize());
  gettimeofday(&t1, NULL);
  calc_time((char *)"Kernel execution time (ms)", t0, t1);


  gettimeofday(&t0, NULL);
  HANDLE_ERROR(cudaMemcpy(h_dist, d_dist, size * sizeof(double), cudaMemcpyDeviceToHost));
  gettimeofday(&t1, NULL);
  calc_time((char *)"GPU → CPU transfer time (ms)", t0, t1);

  HANDLE_ERROR( cudaFree(d_x1) );
  HANDLE_ERROR( cudaFree(d_y1) );
  HANDLE_ERROR( cudaFree(d_x2) );
  HANDLE_ERROR( cudaFree(d_y2) );
  HANDLE_ERROR( cudaFree(d_dist) );

}

PYBIND11_MODULE(haversine_library, m)
{
  m.def("haversine_distance", haversine_distance);
}
