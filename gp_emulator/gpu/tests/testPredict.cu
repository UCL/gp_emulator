#include "gpu_predict_test.h"
#include <time.h>

void testPredict(const real *expX, const real *inputs, const real *invQt, const real *invQ, const real *testing, 
        const real *mu, const real *var, const real *deriv, int M, int N, int D)
{

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
    
    invQ_T = computeTranspose( invQ, M, M );
    inputs_T = computeTranspose( inputs, D, M );
    testing_T = computeTranspose( testing, D, N );

    begin2 = clock();
    predict( expX, inputs_T, invQt, invQ_T, testing_T, gpu_mu, gpu_var, gpu_deriv, N, M, D, D+2);
    end2 = clock();

    gpu_deriv = computeTranspose( gpu_deriv, N, D);
    
    end1 = clock();


//    printf("\n");
//    printf("      -total time including matrix transposing = %f (sec)\n", (double)(end1-begin1)/CLOCKS_PER_SEC);
//    printf("      -pure predict function = %f (sec)\n", (double)(end2-begin2)/CLOCKS_PER_SEC);
    
    compare_result( gpu_mu, mu, N, EPSILON_AVG, EPSILON_MAX, "mu");
    compare_result( gpu_var, var, N, EPSILON_AVG, EPSILON_MAX,  "var");
    compare_result( gpu_deriv, deriv, N * D, EPSILON_AVG, EPSILON_MAX, "deriv");
    
    free(invQ_T);
    free(inputs_T);
    free(testing_T);
    cudaFree(gpu_mu);
    cudaFree(gpu_var);
    cudaFree(gpu_deriv);

    

}


