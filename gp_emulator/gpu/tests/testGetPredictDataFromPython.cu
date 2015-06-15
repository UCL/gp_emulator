#include "gpu_predict_test.h"

/* ==== Set up the methods table ====================== */
static PyMethodDef gpuMethods[] = {
    {"predict_wrap", predict_wrap, METH_VARARGS},
    {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
//Module name must be _C_arraytest in compile and linked 
void init_gpu_predict()  {
	(void) Py_InitModule("_gpu_predict_test", gpuMethods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}



PyObject * testGetPredictDataFromPython(PyObject *self, PyObject *args)
{
    real *c_theta_exp, *c_invQt, *c_invQ;
    real *c_testing, *c_inputs;
    real *c_mu, c_var, c_deriv;
    int N, M, D, theta_size;

    getPredictDataFromPython(args, &c_theta_exp, &c_invQt, &c_invQ,
                             &c_testing, &c_inputs,
                             &c_mu, &c_var, &c_deriv,
                             &N, &M, &D, &theta_size);




}


