#include"gpu_predict_test.h"

void testInitArray(void)
{
    real *c_vec, *d_vec;
    real val;
    int len;
    int i, test, error;
    c_vec = (real *)malloc( int(sizeof(real) * 1e7) );
    cudaMalloc( (void **)&d_vec, int(sizeof(real) * 1e7) );
 
    for( test = 0; test < 6; ++test )
    {
        val = pow( 10.0, test-5);
        len = pow( 10.0, test );
        gpu_init_array(d_vec, val, len);
        cudaMemcpy(d_vec, c_vec, sizeof(real) * len, cudaMemcpyDeviceToHost);
        
        error = 0;
        for( i = 0; i < len; ++i )
        {
            if( ( c_vec[i] - val ) / val > 1e-15 )
                error++;
        }
        CU_ASSERT(error == 0);
    }

    free(c_vec);
    cudaFree(d_vec);
}
