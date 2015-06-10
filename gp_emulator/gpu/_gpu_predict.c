#include "gpu_predict.h"
#include<errno.h>

/*********************************
 * Set up the methods table
 *********************************/
static PyMethodDef gpuMethods[] = 
{
    {"predict_wrap", predict_wrap, METH_VARARGS},
    {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/**********************************
 * Initialize the C_test functions 
 * Module name must be _C_arraytest in compile and linked 
 **********************************/
void init_gpu_predict()  
{
    (void) Py_InitModule("_gpu_predict", gpuMethods);
    import_array();  // Must be present for NumPy.  Called first after above line.
}


/**********************************
 * Check the dimension and data type 
 * of input python arrays.
 **********************************/
void checkRealVector(PyArrayObject *vec)  
{
    if( sizeof( real ) == sizeof( double ) )
    {
        if (vec->descr->type_num != NPY_DOUBLE || vec->nd != 1)  
        {
            printf( "In checkRealVector: array must be of type Double vector\n");
            exit(EXIT_FAILURE);
        }
       
    }
    if( sizeof( real ) == sizeof( float ) ) 
    {
        if (vec->descr->type_num != NPY_FLOAT32 || vec->nd != 1)  
        {
            printf("In checkRealVector: array must be of type Float32 vector\n");
            exit(EXIT_FAILURE);
        }
    }
}



/**********************************
 * Convert data (vector) from PyArrayObject to real
 **********************************/
real *pyvector_to_Carrayptrs(PyArrayObject *arrayin)  
{
    checkRealVector(arrayin);
    return  (real *) arrayin->data;  /* pointer to arrayin data as real */
}



/**********************************
 * getPredictDataFromPython
 * Assign multiple input python array to C array. 
 * Both of them are in 1D. 
 **********************************/
void getPredictDataFromPython(PyObject *args, real **c_theta_exp, real **c_invQt, real **c_invQ, 
                              real **c_testing, real **c_inputs,
                              real **c_mu, real **c_var, real **c_deriv,
                              int *N, int *M, int *D, int *theta_size)
{
    PyArrayObject *py_theta_exp, *py_inputs, *py_invQt, *py_invQ, *py_testing;
    PyArrayObject *py_mu, *py_var, *py_deriv;

    PyArg_ParseTuple ( args, "O!O!O!O!O!O!O!O!iiii",
        &PyArray_Type, &py_theta_exp,
        &PyArray_Type, &py_inputs,
        &PyArray_Type, &py_invQt,
        &PyArray_Type, &py_invQ,
        &PyArray_Type, &py_testing,
        &PyArray_Type, &py_mu,
        &PyArray_Type, &py_var,
        &PyArray_Type, &py_deriv,
        N, M, D, theta_size);

    *c_theta_exp = pyvector_to_Carrayptrs( py_theta_exp);
    *c_inputs = pyvector_to_Carrayptrs( py_inputs );
    *c_invQt = pyvector_to_Carrayptrs( py_invQt );
    *c_invQ = pyvector_to_Carrayptrs( py_invQ );
    *c_testing = pyvector_to_Carrayptrs( py_testing );
    *c_mu = pyvector_to_Carrayptrs( py_mu );
    *c_var = pyvector_to_Carrayptrs( py_var );
    *c_deriv = pyvector_to_Carrayptrs( py_deriv );
}



/**********************************
 * predict_wrap
 * 1) call getPredictDataFromPython;
 * 2) transpose 2D arrays (data arranged in 1D) to column major;
 * 3) calling GPU predict function;
 **********************************/
PyObject *predict_wrap ( PyObject *self, PyObject *args )
{
    int N,M,D,theta_size;
    real *c_theta_exp;
    real *c_inputs, *c_invQt, *c_invQ, *c_testing;
    real *c_mu, *c_var, *c_deriv;
   
    getPredictDataFromPython(args, &c_theta_exp, &c_invQt, &c_invQ,
                             &c_testing, &c_inputs,
                             &c_mu, &c_var, &c_deriv,
                             &N, &M, &D, &theta_size);


    //transpose 2D array to column major to cope with cublas
    real *c_invQ_T, *c_inputs_T, *c_testing_T;
    c_invQ_T = computeTranspose( c_invQ, M, M );
    c_inputs_T = computeTranspose( c_inputs, D, M );
    c_testing_T = computeTranspose( c_testing, D, N);

    predict( c_theta_exp, c_inputs_T, c_invQt, c_invQ_T, c_testing_T, 
            c_mu, c_var, c_deriv,
            N, M, D, theta_size );


    free(c_invQ_T);
    free(c_inputs_T);
    free(c_testing_T);
   
    return Py_BuildValue("");

    
}






















