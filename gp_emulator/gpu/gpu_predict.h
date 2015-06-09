/* A file to test imorting C modules for handling arrays to Python */

#include "Python.h"
#include "arrayobject.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
/* Includes, cuda */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define DOUBLE__PRECISION
#define CUDA_BLOCK 2
 
#ifdef DOUBLE__PRECISION
  #define real double
#else
  #define real float
#endif 



#ifdef DOUBLE__PRECISION
  #define CUBLAS_GEMV cublasDgemv
  #define CUBLAS_GEMM cublasDgemm
  #define CUBLAS_GEAM cublasDgeam
#else 
  #define CUBLAS_GEMV cublasSgemv
  #define CUBLAS_GEMM cublasSgemm
  #define CUBLAS_GEAM cublasSgeam
#endif



PyArrayObject *pyvector(PyObject *objin);
real*pyvector_to_Carrayptrs(PyArrayObject *arrayin);
real **pymatrix_to_Carrayptrs(PyArrayObject *arrayin);
real **ptrvector(long n);
PyObject *predict_wrap ( PyObject *self, PyObject *args );


void getPredictDataFromPython(PyObject *args, real **c_theta_exp, real **c_invQt, real **c_invQ, 
                              real **c_testing, real **c_inputs,
                              real **c_mu, real **c_var, real **c_deriv,
                              int *N, int *M, int *D, int *theta_size);




#ifdef __cplusplus
extern "C"{
#endif
void predict(const real *c_theta_exp,const  real *c_inputs,const real *c_invQt, const real *c_invQ, const real *c_testing, real *c_mu, real *c_var, real *c_deriv, const int N, const int M, const int  D, const int theta_size);
real *computeTranspose(const real *matrix, const  int size_in, const  int size_out);
#ifdef __cplusplus
}
#endif

__global__ void gpu_cdist(const real *input1, const real *input2, real *output, int In1_ld, int In2_ld, int Out_ld);
__global__ void gpu_vectorTimesMatrix(const real *A, const real * v, real *res, int A_ld);
__global__ void gpu_matrixExp(real *matrix, real alpha, real beta);


// error check macros
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

// for CUBLAS V2 API
#define cublasCheckErrors(fn) \
    do { \
        cublasStatus_t __err = fn; \
        if (__err != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "Fatal cublas error: %d (at %s:%d)\n", \
                (int)(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
	    exit(1);\
        } \
    } while (0)

