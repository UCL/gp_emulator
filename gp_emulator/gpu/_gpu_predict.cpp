/**********************************//**
 * source code of share library _gpu_predict.so
 * importing inputs from python
 * calling GPU predict function by predict_wrap()
 * python utility functions are based on the following template:
 *  http://wiki.scipy.org/Cookbook/C_Extensions/NumPy_arrays
 **********************************/

#include "gpu_predict.h"
#include<errno.h>

/*********************************//**
 * Set up the methods table
 *********************************/
static PyMethodDef gpuMethods[] = 
{
    {"predict_wrap", predict_wrap, METH_VARARGS},
    {"pure_c_predict_wrap", pure_c_predict_wrap, METH_VARARGS},
    {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/**********************************//**
 * Initialize the C_test functions 
 * Module name must be _C_arraytest in compile and linked 
 **********************************/
extern "C"{
void init_gpu_predict()  
{
    (void) Py_InitModule("_gpu_predict", gpuMethods);
    import_array();  // Must be present for NumPy.  Called first after above line.
}
}



/**********************************//**
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



/**********************************//**
 * Convert data (vector) from PyArrayObject to real
 **********************************/
real *pyvector_to_Carrayptrs(PyArrayObject *arrayin)  
{
    checkRealVector(arrayin);
    return  (real *) arrayin->data;  /* pointer to arrayin data as real */
}



/**********************************//**
 * getPredictDataFromPython
 * Assign multiple input python arrays, which are all in 1D
 * to C array.
 **********************************/
void getPredictDataFromPython(PyObject *args, real **c_theta_exp, real **c_invQt, real **c_invQ, 
                              real **c_predict, real **c_train,
                              real **c_result, real **c_error, real **c_deriv,
                              int *Npredict, int *Ntrain, int *Ninputs, int *theta_size)
{
    PyArrayObject *py_theta_exp, *py_train, *py_invQt, *py_invQ, *py_predict;
    PyArrayObject *py_result, *py_error, *py_deriv;

    PyArg_ParseTuple ( args, "O!O!O!O!O!O!O!O!iiii",
        &PyArray_Type, &py_theta_exp,
        &PyArray_Type, &py_train,
        &PyArray_Type, &py_invQt,
        &PyArray_Type, &py_invQ,
        &PyArray_Type, &py_predict,
        &PyArray_Type, &py_result,
        &PyArray_Type, &py_error,
        &PyArray_Type, &py_deriv,
        Npredict, Ntrain, Ninputs, theta_size);
    
    *c_theta_exp = pyvector_to_Carrayptrs( py_theta_exp);
    *c_train = pyvector_to_Carrayptrs( py_train );
    *c_invQt = pyvector_to_Carrayptrs( py_invQt ); 
    *c_invQ = pyvector_to_Carrayptrs( py_invQ );
    *c_predict = pyvector_to_Carrayptrs( py_predict );
    *c_result = pyvector_to_Carrayptrs( py_result );
    *c_error = pyvector_to_Carrayptrs( py_error );
    *c_deriv = pyvector_to_Carrayptrs( py_deriv );
}



/**********************************//**
 * predict_wrap
 * 1) call getPredictDataFromPython;
 * 2) transpose 2D arrays (data arranged in 1D) to column major;
 * 3) calling GPU predict function;
 **********************************/
PyObject *predict_wrap ( PyObject *self, PyObject *args )
{
    int i;
    int Npredict, Ntrain, Ninputs, theta_size;
    real *c_theta_exp;
    real *c_train, *c_invQt, *c_invQ, *c_predict;
    real *c_result, *c_error, *c_deriv;
   
    getPredictDataFromPython(args, &c_theta_exp, &c_invQt, &c_invQ,
                             &c_predict, &c_train,
                             &c_result, &c_error, &c_deriv,
                             &Npredict, &Ntrain, &Ninputs,  &theta_size);

    //transpose 2D array to column major to cope with cublas
    real *c_invQ_T, *c_train_T, *c_predict_T;
    c_invQ_T = computeTranspose( c_invQ, Ntrain, Ntrain );
    c_train_T = computeTranspose( c_train, Ninputs, Ntrain );
    c_predict_T = computeTranspose( c_predict, Ninputs, Npredict);

    //get c_theta_exp_sqrt
    real *c_theta_exp_sqrt;
    c_theta_exp_sqrt = (real *)malloc( sizeof(real) * theta_size );
    for( i = 0; i < theta_size; i++ )
    {
        c_theta_exp_sqrt[i] = sqrt( c_theta_exp[i] );
    }

    //predict
    gpuPredict gpu_predict(
        c_theta_exp, c_theta_exp_sqrt,
        c_invQt, c_invQ_T, 
        c_predict_T,c_train_T, 
        c_result, c_error, c_deriv,
        Npredict, Ntrain, Ninputs, theta_size );
    gpu_predict.predict();


    free(c_invQ_T);
    free(c_train_T);
    free(c_predict_T);
    free(c_theta_exp_sqrt);
   
    Py_INCREF(Py_None);
    return Py_None;
}




/**********************************//**
 Pure C implementation for benchmarking purpose only. 
 **********************************/
PyObject *pure_c_predict_wrap ( PyObject *self, PyObject *args )
{
    int i;
    int Npredict, Ntrain, Ninputs, theta_size;
    real *c_theta_exp;
    real *c_train, *c_invQt, *c_invQ, *c_predict;
    real *c_result, *c_error, *c_deriv;
   
    getPredictDataFromPython(args, &c_theta_exp, &c_invQt, &c_invQ,
                             &c_predict, &c_train,
                             &c_result, &c_error, &c_deriv,
                             &Npredict, &Ntrain, &Ninputs,  &theta_size);

    //transpose 2D array to column major to cope with cublas
    real *c_invQ_T, *c_train_T, *c_predict_T;
    c_invQ_T = computeTranspose( c_invQ, Ntrain, Ntrain );
    c_train_T = computeTranspose( c_train, Ninputs, Ntrain );
    c_predict_T = computeTranspose( c_predict, Ninputs, Npredict);

    //get c_theta_exp_sqrt
    real *c_theta_exp_sqrt;
    c_theta_exp_sqrt = (real *)malloc( sizeof(real) * theta_size );
    for( i = 0; i < theta_size; i++ )
    {
        c_theta_exp_sqrt[i] = sqrt( c_theta_exp[i] );
    }

    //predict
    pureCPredict CPredict(
        c_theta_exp, c_theta_exp_sqrt,
        c_invQt, c_invQ_T, 
        c_predict_T,c_train_T, 
        c_result, c_error, c_deriv,
        Npredict, Ntrain, Ninputs, theta_size );
    
    CPredict.predict();


    free(c_invQ_T);
    free(c_train_T);
    free(c_predict_T);
    free(c_theta_exp_sqrt);
   
    Py_INCREF(Py_None);
    return Py_None;
}



















