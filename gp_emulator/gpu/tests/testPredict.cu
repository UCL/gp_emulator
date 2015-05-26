#include "gpu_predict_test.h"
#include <time.h>

void testPredict(const real *expX, const real *inputs, const real *invQt, const real *invQ, const real *testing, 
        const real *mu, const real *var, const real *deriv, int M, int N, int D)
{

    int i;
    real e;

    clock_t begin1, begin2, end1, end2;

    real *gpu_mu, *gpu_var, *gpu_deriv;
    gpu_mu = (real *)malloc(sizeof(real) * N);
    gpu_var = (real *)malloc(sizeof(real) * N);
    gpu_deriv = (real *)malloc(sizeof(real) * N * D);


    real *invQ_T, *inputs_T, *testing_T;
    invQ_T = (real *)malloc(sizeof(real) * M * M);
    inputs_T = (real *)malloc(sizeof(real) * N * D);
    testing_T = (real *)malloc(sizeof(real) * N * D);
   
    

    begin1 = clock();

    for( i = 0; i < M * M; ++i )
        invQ_T[i] = invQ[i];
    for( i = 0; i < N * D; ++i )
        inputs_T[i] = inputs[i];
    for( i = 0; i < N * D; ++i )
        testing_T[i] = testing[i];



    
    computeTranspose( invQ_T, M, M );
    computeTranspose( inputs_T, D, M );
    computeTranspose( testing_T, D, N );

    begin2 = clock();
    predict( expX, inputs_T, invQt, invQ_T, testing_T, gpu_mu, gpu_var, gpu_deriv, N, M, D, D+2);
    end2 = clock();

    computeTranspose( gpu_deriv, N, D);
    
    end1 = clock();


    printf("\n");
    printf("      -total time including matrix transposing = %f (sec)\n", (double)(end1-begin1)/CLOCKS_PER_SEC);
    printf("      -pure predict function = %f (sec)\n", (double)(end2-begin2)/CLOCKS_PER_SEC);
    for( i = 0; i < N * D; ++i )
    {
        e = abs( gpu_deriv[i] - deriv[i] );
        CU_ASSERT( e < 1e-6 );
    }

    for( i = 0; i < N; ++i )
    {
        e = abs( gpu_mu[i] - mu[i] );
        CU_ASSERT( e < 1e-6);
    }

    for( i = 0; i < N; ++i)
    {
        e = abs( gpu_var[i] - var[i] );
        CU_ASSERT( e < 1e-6 );
    }


        
    free(invQ_T);
    free(inputs_T);
    free(testing_T);
    cudaFree(gpu_mu);
    cudaFree(gpu_var);
    cudaFree(gpu_deriv);

    

}


