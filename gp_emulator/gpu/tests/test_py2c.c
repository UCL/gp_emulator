
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





/* ==== Create 1D Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.             */
real *pyvector_to_Carrayptrs(PyArrayObject *arrayin)  {
    int i,n;
    //not_realvector(arrayin);    
    n=arrayin->dimensions[0];
    return  (real *) arrayin->data;  /* pointer to arrayin data as double */
}




PyObject *predict_wrap ( PyObject *self, PyObject *args )
{
    int N,M,D,theta_size;
    PyArrayObject *py_theta_exp, *py_inputs, *py_invQt, *py_invQ, *py_testing;
    PyArrayObject *py_mu, *py_var, *py_deriv;
    real *c_theta_exp;
    real *c_inputs, *c_invQt, *c_invQ, *c_testing;
    real *c_mu, *c_var, *c_deriv;


    PyArg_ParseTuple ( args, "O!O!O!O!O!O!O!O!iiii",
        &PyArray_Type, &py_theta_exp,
        &PyArray_Type, &py_inputs,
        &PyArray_Type, &py_invQt,
        &PyArray_Type, &py_invQ,
        &PyArray_Type, &py_testing,
        &PyArray_Type, &py_mu,
        &PyArray_Type, &py_var,
        &PyArray_Type, &py_deriv,
        &N, &M, &D,&theta_size);

    c_theta_exp = pyvector_to_Carrayptrs( py_theta_exp);
    c_inputs = pyvector_to_Carrayptrs( py_inputs );
    c_invQt = pyvector_to_Carrayptrs( py_invQt );
    c_invQ = pyvector_to_Carrayptrs( py_invQ );
    c_testing = pyvector_to_Carrayptrs( py_testing );
    c_mu = pyvector_to_Carrayptrs( py_mu );
    c_var = pyvector_to_Carrayptrs( py_var );
    c_deriv = pyvector_to_Carrayptrs( py_deriv );

    //transpose 2D array to column major to cope with cublas
    real *c_invQ_T, *c_inputs_T, *c_testing_T;
    c_invQ_T = computeTranspose( c_invQ, M, M );
    c_inputs_T = computeTranspose( c_inputs, D, M );
    c_testing_T = computeTranspose( c_testing, D, N );
    
    predict( c_theta_exp, c_inputs_T, c_invQt, c_invQ_T, c_testing_T, 
            c_mu, c_var, c_deriv,
            N, M, D, theta_size );
    int ii;
    for(ii = 0;ii<10;ii++)
        printf("%f|",c_testing[ii]);

    free(c_invQ_T);
    free(c_inputs_T);
    free(c_testing_T);
    
    return Py_BuildValue("");

    
}






















