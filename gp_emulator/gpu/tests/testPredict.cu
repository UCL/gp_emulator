#include "gpu_predict_test.h"
#include <time.h>

void testPredict(void)
{
    real *gpu_result, *gpu_error, *gpu_deriv;
    gpu_result = (real *)malloc(sizeof(real) * Npredict_t);
    gpu_error = (real *)malloc(sizeof(real) * Npredict_t);
    gpu_deriv = (real *)malloc(sizeof(real) * Npredict_t * Ninputs_t);

    real *invQ_T, *train_T, *predict_T;
    invQ_T = (real *)malloc(sizeof(real) * Ntrain_t * Ntrain_t);
    train_T = (real *)malloc(sizeof(real) * Npredict_t * Ninputs_t);
    predict_T = (real *)malloc(sizeof(real) * Npredict_t * Ninputs_t);

    
    invQ_T = computeTranspose( invQ, Ntrain_t, Ntrain_t );
    train_T = computeTranspose( in_train, Ninputs_t, Ntrain_t );
    predict_T = computeTranspose( in_predict, Ninputs_t, Npredict_t );

    gpuPredict gpu_predict(expX, expXsqrt, invQt, invQ_T, predict_T, train_T,
            gpu_result, gpu_error, gpu_deriv, Npredict_t, Ntrain_t, Ninputs_t, theta_size_t);
    gpu_predict.predict();

    gpu_deriv = computeTranspose( gpu_deriv, Npredict_t, Ninputs_t);
    
    compare_result( gpu_result, result_py, Npredict_t, EPSILON_AVG, EPSILON_MAX, "result");
    compare_result( gpu_error, error_py, Npredict_t, EPSILON_AVG, EPSILON_MAX,  "error");
    compare_result( gpu_deriv, deriv_py, Npredict_t * Ninputs_t, EPSILON_AVG, EPSILON_MAX, "deriv");
    
    free(invQ_T);
    free(train_T);
    free(predict_T);
    cudaFree(gpu_result);
    cudaFree(gpu_error);
    cudaFree(gpu_deriv);
}


