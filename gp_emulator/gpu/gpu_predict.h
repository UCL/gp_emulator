/* A file to test importing C modules for handling arrays to Python */

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

#define CUDA_BLOCK 2
#define MIN_NPREDICT 1000 //ensure there will be enough amount of threads to launch. 
#define MAX_NUM_THREAD 1024 //sm_20
#define MAX_NUM_BLOCK 65536

#ifdef DOUBLE__PRECISION
  #define real double
#else
  #define real float
#endif 

#ifdef DOUBLE__PRECISION
  #define CUBLAS_GEMV cublasDgemv
  #define CUBLAS_GEMM cublasDgemm
  #define CUBLAS_GEAM cublasDgeam
  #define BLAS_GEMV   cblas_dgemv
  #define BLAS_GEMM   cblas_dgemm
#else 
  #define CUBLAS_GEMV cublasSgemv
  #define CUBLAS_GEMM cublasSgemm
  #define CUBLAS_GEAM cublasSgeam
  #define BLAS_GEMV   cblas_sgemv
  #define BLAS_GEMM   cblas_sgemm
#endif

__forceinline__ __host__ __device__
int IDX2D(int row, int col, int lead_dim)
{
    return(((col)*(lead_dim))+(row));
}

PyArrayObject *pyvector(PyObject *objin);
real*pyvector_to_Carrayptrs(PyArrayObject *arrayin);
real **pymatrix_to_Carrayptrs(PyArrayObject *arrayin);
real **ptrvector(long n);
PyObject *predict_wrap ( PyObject *self, PyObject *args );
PyObject *pure_c_predict_wrap ( PyObject *self, PyObject *args );

void getPredictDataFromPython(PyObject *args, real **c_theta_exp, real **c_invQt, real **c_invQ, 
                              real **c_testing, real **c_inputs,
                              real **c_mu, real **c_var, real **c_deriv,
                              int *N, int *M, int *D, int *theta_size);

#ifdef __cplusplus
extern "C"{
#endif
real *computeTranspose(const real *matrix, const  int size_in, const  int size_out);
void gpu_vectorTimesMatrix(const real *A, const real *v, real *res, int nrows, int ncols);
void gpu_init_array(real *vec, const int init_val, const int vec_len);
void gpu_cdist(const real *input1, const real *input2, real *output, const int nrow1, const int ncol1,
        const int nrow2, const int ncol2);
void gpu_matrixExp( real *matrix,const real alpha,const real beta, const int size );
void gpu_elementwiseMult( const real *v1, real *v2, const int size );
void gpu_scalarMinusVec( real *matrix, const real scalar, const int size );
real* gpu_rowSum(const real *A, const int A_nrows,const int A_ncols);
void gpu_crossMinus(const real *v1, const real *v2, real *mat_res, const int v1_len, const int v2_len);
#ifdef __cplusplus
}
#endif

class gpuPredict
{
  real *c_result, *c_error, *c_deriv;
  const real *c_theta_exp, *c_theta_exp_sqrt;
  const real *c_invQt, *c_invQ, *c_predict, *c_train, *c_inputs;
  const int Npredict, Ntrain, Ninputs;
  const int theta_size;

  real *d_result, *d_error, *d_deriv;
  real *d_theta_exp, *d_theta_exp_sqrt;
  real *d_invQt, *d_invQ, *d_predict, *d_train, *d_inputs;
  real *d_dist_matrix; real *d_dist_matrix_T;
  cublasHandle_t handle;
  public:
    gpuPredict(real *ctheta_exp, real *ctheta_exp_sqrt, real *cinvQt,
              real *cinvQ, real *cpredict, real *ctrain, real *cresult, 
              real *cerror, real *cderiv, int npredict, int ntrain,
              int ninputs, int thetasize):
          c_theta_exp(ctheta_exp), c_theta_exp_sqrt(ctheta_exp_sqrt),  c_invQt(cinvQt),
          c_invQ(cinvQ), c_predict(cpredict), c_train(ctrain), c_result(cresult), c_error(cerror),   
          c_deriv(cderiv), Npredict(npredict), Ntrain(ntrain), Ninputs(ninputs), theta_size(thetasize){};
    
    real * gpu_transpose(real *, const int, const int);
    void init_gpu(void);
    void compute_distance(void);
    void compute_result(void);
    void compute_error(void);
    void compute_deriv(void);
    void predict(void);
    void free_gpu(void);
};

class pureCPredict 
{
  real *c_result, *c_error, *c_deriv;
  const real *c_theta_exp, *c_theta_exp_sqrt;
  real *c_invQt, *c_invQ, *c_predict, *c_train, *c_inputs;
  real * c_dist_matrix, *c_dist_matrix_T;
  const int Npredict, Ntrain, Ninputs;
  const int theta_size;

public:
  pureCPredict(real *ctheta_exp, real *ctheta_exp_sqrt, real *cinvQt,
              real *cinvQ, real *cpredict, real *ctrain, real *cresult, 
              real *cerror, real *cderiv, int npredict, int ntrain,
              int ninputs, int thetasize):
          c_theta_exp(ctheta_exp), c_theta_exp_sqrt(ctheta_exp_sqrt),  c_invQt(cinvQt),
          c_invQ(cinvQ), c_predict(cpredict), c_train(ctrain), c_result(cresult), c_error(cerror),   
          c_deriv(cderiv), Npredict(npredict), Ntrain(ntrain), Ninputs(ninputs), theta_size(thetasize){};
  void predict(void);
  void compute_result(void);
  void compute_error(void);
  void compute_deriv(void);
};






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

