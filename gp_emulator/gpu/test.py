#!/usr/bin/python
import numpy as np
import scipy.spatial.distance as dist
import _gpu_predict
import time

class GP:
    def __init__ (self, P, N, M):
        self.D = P
        self.theta = np.random.random((P+2))
        self.inputs = np.random.random ((M,P))
        self.invQ = np.random.random((M,M))
        self.invQt = np.random.random ((M))

#    @profile
    def predict ( self, testing, do_unc=True ):
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
            #print deriv[0:10,d]

        if do_unc:
            return mu, var, deriv
        else:
	    return mu, deriv

    def get_testing_val (self, testing, do_unc = True):
        
        ( nn, D ) = testing.shape
        assert D == self.D
        expX=np.exp(self.theta)


        
        global t_expXsqrt, t_inputs, t_testing
        t_expX = expX
        t_expXsqrt = np.sqrt(expX[:(self.D)])
        t_inputs = self.inputs
        t_testing = testing
        t_invQ = self.invQ

        global cdist_test_var_1 ,cdist_test_var_2, cdist_test_var_3
        cdist_test_var_1= t_expXsqrt * t_inputs #np.sqrt(expX[:(self.D)])*self.inputs
        cdist_test_var_2 = np.sqrt(expX[:(self.D)])*testing
        cdist_test_var_3 = dist.cdist ( np.sqrt(expX[:(self.D)])*self.inputs, np.sqrt(expX[:(self.D)])*testing, 'sqeuclidean')
        cdist_test_var_4 = expX[self.D]*np.exp(-0.5*cdist_test_var_3)

        mu = np.dot( cdist_test_var_4.T, self.invQt)


        var_test_1 = np.dot(self.invQ, cdist_test_var_4)
        
        
        
        t_expX.tofile("./tests/data/set_%d_%d_%d/expX.bin"%(N,M,P))
        t_expXsqrt.tofile("./tests/data/set_%d_%d_%d/expXsqrt.bin"%(N,M,P))
        t_inputs.tofile("./tests/data/set_%d_%d_%d/inputs.bin"%(N,M,P))
        t_testing.tofile("./tests/data/set_%d_%d_%d/testing.bin"%(N,M,P))
        cdist_test_var_1.tofile("./tests/data/set_%d_%d_%d/cdist_test_var1.bin"%(N,M,P))
        cdist_test_var_2.tofile("./tests/data/set_%d_%d_%d/cdist_test_var2.bin"%(N,M,P))
        cdist_test_var_3.tofile("./tests/data/set_%d_%d_%d/cdist_a.bin"%(N,M,P))
        cdist_test_var_4.tofile("./tests/data/set_%d_%d_%d/cdist_expa.bin"%(N,M,P))    
        
        t_invQ.tofile("./tests/data/set_%d_%d_%d/invQ.bin"%(N,M,P))
        mu.tofile("./tests/data/set_%d_%d_%d/mu.bin"%(N,M,P))
        var_test_1.tofile("./tests/data/set_%d_%d_%d/var_test1.bin"%(N,M,P))
        self.invQt.tofile("./tests/data/set_%d_%d_%d/invQt.bin"%(N,M,P))


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
            print deriv[0:10,d]

        var.tofile("./tests/data/set_%d_%d_%d/var.bin"%(N,M,P))
        deriv.tofile("./tests/data/set_%d_%d_%d/deriv.bin"%(N,M,P))

        if do_unc:
            return mu, var, deriv
        else:
	    return mu, deriv



    def gpu_predict ( self, testing, do_unc = True ):# self, testing, do_unc=True):
        '''GPU predict function
        '''
        ( nn, D ) = testing.shape
        assert D == self.D
        expX=np.exp(self.theta)

        N=testing.shape[0]
        M=self.inputs.shape[0]
        theta_size=self.theta.size 
        #print 'sqrt(expX[:(self.D)])*self.inputs=\n', test_var_1
        #print 'sqrt(expX)*testing\n', test_var_2
        #print 'a_ =\n', test_var_3

        a =_gpu_predict.predict_wrap(
            expX,
            self.inputs,
            self.invQt,
            self.invQ,
            testing,
            N,M,D,theta_size)
        #print 'testing\n',testing
        #print 'a\n',a


        # for passing the test (temp)
        mu=0
        var=0
        deriv=0
        if do_unc:
            return mu, var, deriv
        else:
            return mu, deriv


if __name__ == '__main__':
    N=1.5e5
    M=250
    P=10
    testing=np.random.random((N,P))
    gp=GP(P,N,M)
    
    start = time.time()
    gp.predict(testing)
    end = time.time()
    cputime = end -start
    print "cpu time", end - start
#    gp.get_testing_val(testing)
    start = time.time()
    gp.gpu_predict(testing)
    end =time.time()
    gputime = end - start
    print "gpu time", end - start
    print "speedup:", cputime/gputime, 'x'
    #write testing set
   
