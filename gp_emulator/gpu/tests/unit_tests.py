import numpy as np
import scipy.spatial.distance as dist
import time
import os as os
class GP:
    def __init__ (self, P, N, M):
        self.D = P
        self.theta = np.random.random((P+2))
        self.inputs = np.random.random ((M,P))
        self.invQ = np.random.random((M,M))
        self.invQt = np.random.random ((M))

    def get_testing_val (self, testing, do_unc = True):
        
        ( nn, D ) = testing.shape
        assert D == self.D
        expX=np.exp(self.theta)


        
        t_expX = expX
        t_expXsqrt = np.sqrt(expX[:(self.D)])
        t_inputs = self.inputs
        t_testing = testing
        t_invQ = self.invQ

        cdist_test_var_1= t_expXsqrt * t_inputs #np.sqrt(expX[:(self.D)])*self.inputs
        cdist_test_var_2 = np.sqrt(expX[:(self.D)])*testing
        cdist_test_var_3 = dist.cdist ( np.sqrt(expX[:(self.D)])*self.inputs, np.sqrt(expX[:(self.D)])*testing, 'sqeuclidean')
        cdist_test_var_4 = expX[self.D]*np.exp(-0.5*cdist_test_var_3)

        mu = np.dot( cdist_test_var_4.T, self.invQt)


        var_test_1 = np.dot(self.invQ, cdist_test_var_4)
        
        
        
        np.float32(t_expX).tofile("./data/expX.bin")
        np.float32(t_expXsqrt).tofile("./data/expXsqrt.bin")
        np.float32(t_inputs).tofile("./data/inputs.bin")
        np.float32(t_testing).tofile("./data/testing.bin")
        np.float32(cdist_test_var_1).tofile("./data/cdist_test_var1.bin")
        np.float32(cdist_test_var_2).tofile("./data/cdist_test_var2.bin")
        np.float32(cdist_test_var_3).tofile("./data/cdist_a.bin")
        np.float32(cdist_test_var_4).tofile("./data/cdist_expa.bin")    
        
        np.float32(t_invQ).tofile("./data/invQ.bin")
        np.float32(mu).tofile("./data/mu.bin")
        np.float32(var_test_1).tofile("./data/var_test1.bin")
        np.float32(self.invQt).tofile("./data/invQt.bin")


        ( nn, D ) = testing.shape
        assert D == self.D

        expX = np.exp ( self.theta )
              
        a = dist.cdist ( np.sqrt(expX[:(self.D)])*self.inputs, np.sqrt(expX[:(self.D)])*testing, 'sqeuclidean')
        a = expX[self.D]*np.exp(-0.5*a)
        b = expX[self.D]
        
        mu = np.dot( a.T, self.invQt)
        if do_unc:
            var = b - np.sum (  a * np.dot(self.invQ,a), axis=0)
        # Derivative and partial derivatives of the function
        deriv = np.zeros ( ( nn, self.D ) )

        for d in xrange ( self.D ):
            aa = self.inputs[:,d].flatten()[None,:] - testing[:,d].flatten()[:,None]
            
            c = a*aa.T
            deriv[:, d] = expX[d]*np.dot(c.T, self.invQt)
#            print deriv[0:10,d]

        np.float32(var).tofile("./data/var.bin")
        np.float32(deriv).tofile("./data/deriv.bin")

        if do_unc:
            return mu, var, deriv
        else:
	    return mu, deriv


if __name__ == '__main__':
    N=8e4
    M=250
    P=10
    testing=np.random.random((N,P))
    gp=GP(P,N,M)
    
    gp.get_testing_val(testing)
    command = "./gpu_predict_test  %d %d %d" % (M, N, P)
    os.system(command)
   
