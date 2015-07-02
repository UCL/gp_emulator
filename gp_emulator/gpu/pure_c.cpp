#include"gpu_predict.h"
#include <gsl/gsl_cblas.h>
#include<time.h>
void cpu_vectorTimesMatrix(const real *matrix, const real *vector, real *res, int lead_dim, int second_dim)
{
    int ix, iy;
    for( ix = 0 ; ix < lead_dim; ++ix )
    {
        for( iy = 0; iy < second_dim; ++iy)
        {
                res[IDX2D(ix, iy, lead_dim)] = matrix[IDX2D(ix, iy, lead_dim)] * vector[iy];
        }
    }
}


void cpu_cdist(const real *input1, const real *input2, real *output, const int nrow1, const int ncol1, const int nrow2, const int ncol2)
{
    int i, ix, iy, iz;
    for( i = 0; i < nrow1 * nrow2; ++i)
        output[i] = 0;
    for( ix = 0; ix < nrow2; ++ix )
    {
        for( iy = 0; iy < nrow1; ++iy )
        {
            for( iz = 0; iz < ncol1; ++iz )
            {
                output[IDX2D(ix, iy, nrow2)] += (input1[IDX2D(iy, iz, nrow1)] - input2[IDX2D(ix, iz, nrow2)]) * 
                    (input1[IDX2D(iy, iz, nrow1)] - input2[IDX2D(ix, iz, nrow2)]);
            }
        }
    }
}

void pureCPredict::predict(void)
{
   compute_result();
   compute_error();
   compute_deriv();
}



void pureCPredict::compute_result(void)
{
    int i;
    clock_t start, diff;
    real *c_res_mv1, *c_res_mv2;
    c_res_mv1 = (real *)malloc(sizeof(real) * Ntrain * Ninputs);
    c_res_mv2 = (real *)malloc(sizeof(real) * Npredict * Ninputs);
    c_dist_matrix = (real *)malloc(sizeof(real) * Npredict * Ntrain);
    cpu_vectorTimesMatrix(c_train, c_theta_exp_sqrt, c_res_mv1, Ntrain, Ninputs);
    cpu_vectorTimesMatrix(c_predict, c_theta_exp_sqrt, c_res_mv2, Npredict, Ninputs);
start = clock();
    cpu_cdist(c_res_mv1, c_res_mv2, c_dist_matrix, Ntrain, Ninputs, Npredict, Ninputs);
diff = clock() - start;
int msec = diff * 1000 / CLOCKS_PER_SEC;
printf("cdist:  %f\n",float(msec)/1000);

    for( i = 0; i < Ntrain * Npredict; ++i)
        c_dist_matrix[i] = c_theta_exp[Ninputs] * exp( -0.5 * c_dist_matrix[i]);
    BLAS_GEMV(CblasColMajor, CblasNoTrans, Npredict, Ntrain, 1.0, c_dist_matrix, Npredict, c_invQt, 1, 0.0, c_result, 1);
    free(c_res_mv1);
    free(c_res_mv2);
}





void pureCPredict::compute_error(void)
{
    int i;
    real *result_dot, *vec_one;
    result_dot = (real *)malloc( sizeof(real) * Ntrain * Npredict);
    vec_one = (real *)malloc( sizeof(real) * Npredict);
    BLAS_GEMM(CblasColMajor, CblasNoTrans, CblasTrans, Ntrain, Npredict, Ntrain, 1.0, 
            c_invQ, Ntrain, c_dist_matrix, Npredict, 0.0, result_dot, Ntrain);
    c_dist_matrix_T = computeTranspose(c_dist_matrix, Npredict, Ntrain);
    for( i = 0; i < Npredict * Ntrain; ++i )
        result_dot[i] = c_dist_matrix_T[i] * result_dot[i];
    for( i = 0; i < Npredict; ++i)
    {
        c_error[i] = 0.0;
        vec_one[i] = 1.0;
    }

    BLAS_GEMV(CblasColMajor, CblasTrans, Ntrain, Npredict, 
            1.0, result_dot, Ntrain, vec_one, 1, 0.0, c_error, 1);

    for( i = 0; i < Npredict; ++i){
        c_error[i] = c_theta_exp[Ninputs] - c_error[i];
    }
    free(c_dist_matrix);
    free(result_dot);
    free(vec_one);
}



void pureCPredict::compute_deriv(void)
{
    int i, j, n;
    clock_t start, end, diff;
    diff=0;
    real *aa;
    real *ptr_train, *ptr_predict, *ptr_deriv;
    ptr_train = c_train;
    ptr_predict = c_predict;
    ptr_deriv = c_deriv;
    aa = (real *)malloc( sizeof(real) * Ntrain * Npredict );
    real alpha;
    for( n = 0; n < Ninputs; ++n)
    {
        for( i = 0; i < Ntrain; ++i )
        {
            for( j = 0; j < Npredict; ++j )
            {
                aa[IDX2D(i, j, Ntrain)] = ptr_train[i] - ptr_predict[j];
            }
        }
        ptr_train = ptr_train + Ntrain;
        ptr_predict = ptr_predict + Npredict;

        start = clock();
        for(i = 0; i < Ntrain * Npredict; ++i )
        {
            aa[i] = c_dist_matrix_T[i] * aa[i];
        }
        end = clock();
        diff += end - start;

        alpha = c_theta_exp[n];
        BLAS_GEMV(CblasColMajor, CblasTrans, Ntrain, Npredict, alpha, aa, Ntrain, c_invQt, 1, 0, ptr_deriv, 1);
        ptr_deriv = ptr_deriv + Npredict;
    }
    printf("mat_ele2 %f\n", (float)(diff/CLOCKS_PER_SEC));
    
   free(aa);
}





