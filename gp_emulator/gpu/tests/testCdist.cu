#include "gpu_predict_test.h"
#define IDX2D(i,j,ld) (((j)*(ld))+(i))


void testCdist(const real *in1,const real *in2, const real *res, const int in1_nrows, const int in2_nrows, const int in_ncols,  const dim3 nblocks, const dim3 nthreads)
{
    int i, j, k, error;
    real *in1_T, *in2_T, *gpu_res;
    real *d_in1, *d_in2, *d_res; 

    in1_T = (real *)malloc( sizeof(real) * in1_nrows * in_ncols );
    in2_T = (real *)malloc( sizeof(real) * in2_nrows * in_ncols );
    gpu_res = (real *)malloc( sizeof(real) * in1_nrows * in2_nrows ); 
    for( i = 0; i < in1_nrows * in_ncols; i++ )
        in1_T[i] = in1[i];

    for( i = 0; i < in2_nrows * in_ncols; i++ )
        in2_T[i] = in2[i];

    computeTranspose( in1_T, in_ncols, in1_nrows );
    computeTranspose( in2_T, in_ncols, in2_nrows );
    
    
    
    /*GPU part*/
    for( i = 0; i < in1_nrows * in2_nrows; i++ )
        gpu_res[i] = 0;
    
    
    cudaMalloc((void **)&d_in1, sizeof(real) * in1_nrows * in_ncols );
    cudaMalloc((void **)&d_in2, sizeof(real) * in2_nrows * in_ncols );
    cudaMalloc((void **)&d_res, sizeof(real) * in2_nrows * in1_nrows );
    cublasCheckErrors(cublasSetMatrix( in1_nrows, in_ncols, sizeof(real), in1_T, in1_nrows, d_in1, in1_nrows) );
    cublasCheckErrors(cublasSetMatrix( in2_nrows, in_ncols, sizeof(real), in2_T, in2_nrows, d_in2, in2_nrows) );
    cublasCheckErrors(cublasSetMatrix( in2_nrows, in1_nrows, sizeof(real), gpu_res, in2_nrows, d_res,in2_nrows) );
    gpu_cdist <<< nblocks, nthreads  >>>(d_in1, d_in2, d_res, in1_nrows, in2_nrows, in2_nrows, in_ncols);
    cudaMemcpy(gpu_res, d_res, sizeof(real) * in2_nrows * in1_nrows, cudaMemcpyDeviceToHost);
    
    error = 0;
    for( i = 0; i < in1_nrows * in2_nrows; i++ )
    {
         if( abs( res[i] - gpu_res[i] ) > epsilon )
             error++;
    }
    if( error != 0)
        printf( "  cdist: [%d/%d]\n", error, in1_nrows*in2_nrows );
 
   CU_ASSERT( error == 0 );

   free(in1_T);
   free(in2_T);
   free(gpu_res);
   
   cudaFree(d_in1);
   cudaFree(d_in2);
   cudaFree(d_res);
}
