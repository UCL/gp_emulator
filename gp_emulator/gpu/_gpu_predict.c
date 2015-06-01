
#include "gpu_predict.h"

/* ==== Set up the methods table ====================== */
static PyMethodDef gpuMethods[] = {
    {"predict_wrap", predict_wrap, METH_VARARGS},
	{NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
//Module name must be _C_arraytest in compile and linked 
void init_gpu_predict()  {
	(void) Py_InitModule("_gpu_predict", gpuMethods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}






/* #### Vector Utility functions ######################### */

/* ==== Make a Python Array Obj. from a PyObject, ================
     generates a double vector w/ contiguous memory which may be a new allocation if
     the original was not a double type or contiguous 
  !! Must DECREF the object returned from this routine unless it is returned to the
     caller of this routines caller using return PyArray_Return(obj) or
     PyArray_BuildValue with the "N" construct   !!!
*/
PyArrayObject *pyvector(PyObject *objin)  {
    if( sizeof( real ) == sizeof( double ) )
        return (PyArrayObject *) PyArray_ContiguousFromObject(objin,NPY_DOUBLE, 1,1);
    if( sizeof( real ) == sizeof( float ) )
        return (PyArrayObject *) PyArray_ContiguousFromObject(objin,NPY_FLOAT32, 1,1);

}
/* ==== Create 1D Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.             */
real *pyvector_to_Carrayptrs(PyArrayObject *arrayin)  {
    int i,n;
    
    n=arrayin->dimensions[0];
    return (real *) arrayin->data;  /* pointer to arrayin data as double */
}
/* ==== Check that PyArrayObject is a double (Float) type and a vector ==============
    return 1 if an error and raise exception */ 
int  not_realvector(PyArrayObject *vec)  {
    if( sizeof( real ) == sizeof( double ) )
    {
        if (vec->descr->type_num != NPY_DOUBLE || vec->nd != 1)  
        {
            PyErr_SetString(PyExc_ValueError, "In not_realvector: array must be of type Float and 1 dimensional (n).");
            return 1;  
        }
        return 0;
    }
    if( sizeof( real ) == sizeof( float ) ) 
    {
        if (vec->descr->type_num != NPY_FLOAT32 || vec->nd != 1)  
        {
            PyErr_SetString(PyExc_ValueError, "In not_realvector: array must be of type Float and 1 dimensional (n).");
            return 1;  
        }
        return 0;
    }
 
}



/* ==== Create Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.
    Memory is allocated!                                    */

real **ptrvector(long n)  {
    real **v;
    v=(real **)malloc((size_t) (n*sizeof(real)));
    if (!v)   {
        printf("In **ptrvector. Allocation of memory for real array failed.");
        exit(0);  }
    return v;
}


real **pymatrix_to_Carrayptrs(PyArrayObject *arrayin)  {
    real **c, *a;
    int i,n,m;
    
    n=arrayin->dimensions[0];
    m=arrayin->dimensions[1];
    c=ptrvector(n);
    a=(real *) arrayin->data;  /* pointer to arrayin data as double */
    for ( i=0; i < n; i++)  {
        c[i]=a+i*m;  }
    return c;
}




PyObject *predict_wrap ( PyObject *self, PyObject *args )
{
    int N,M,D,theta_size;
    
    PyArrayObject *py_theta_exp, *py_inputs, *py_invQt, *py_invQ, *py_testing;
    PyArrayObject * temp;
    real *c_theta_exp;
    real *c_inputs, *c_invQt, *c_invQ, *c_testing;
    real *c_mu, *c_var, *c_deriv;


    PyArg_ParseTuple ( args, "O!O!O!O!O!iiii",
        &PyArray_Type, &py_theta_exp,
        &PyArray_Type, &py_inputs,
        &PyArray_Type, &py_invQt,
        &PyArray_Type, &py_invQ,
        &PyArray_Type, &py_testing,
        &N, &M, &D,&theta_size);


    

    c_theta_exp = pyvector_to_Carrayptrs( py_theta_exp);
    c_inputs = pyvector_to_Carrayptrs( py_inputs );
    c_invQt = pyvector_to_Carrayptrs( py_invQt );
    c_invQ = pyvector_to_Carrayptrs( py_invQ );
    c_testing = pyvector_to_Carrayptrs( py_testing );

    c_mu = (real *)malloc(sizeof(real) * N);
    c_var = (real *)malloc(sizeof(real) * N);
    c_deriv = (real *)malloc(sizeof(real) * N * D);
    //transpose 2D array to column major to cope with cublas
    computeTranspose( c_invQ, M, M );
    computeTranspose( c_inputs, D, M );
    computeTranspose( c_testing, D, N );

    predict( c_theta_exp, c_inputs, c_invQt, c_invQ, c_testing, 
            c_mu, c_var, c_deriv,
            N, M, D, theta_size );

    
    /*transpose back to row major
    computeTranspose( c_testing, N, D );
    computeTranspose( c_invQ, M, D );
    computeTranspose( c_inputs, D, M );
    */
    


    return Py_BuildValue ( "O", PyArray_Return ( py_testing ) );
}






















