#!/usr/local/bin/python
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


    def gpu_predict ( self, testing, do_unc = True ):# self, testing, do_unc=True):
        '''GPU predict function
        '''
        ( nn, D ) = testing.shape
        assert D == self.D
        expX=np.exp(self.theta)

        N=testing.shape[0]
        M=self.inputs.shape[0]
        theta_size=self.theta.size 
        
        mu = np.float32(np.zeros(N))
        var = np.float32(np.zeros(N))
        deriv = np.float32(np.zeros((N,D)))

        a = _gpu_predict.predict_wrap(
                np.float32(expX),
                np.float32(self.inputs),
                np.float32(self.invQt),
                np.float32(self.invQ),
                np.float32(testing),
                mu, var, deriv,
                N, M, D, theta_size)
        if do_unc:
            return mu, var, deriv
        else:
            return mu, deriv


if __name__ == '__main__':
    for i in xrange(1):#xrange(1000, 1800000, 5000):
    	N=1e5
    	M=250
    	P=10
    	testing=np.random.random((N,P))
    	gp=GP(P,N,M)
    
    	start = time.time()
    	#gp.predict(testing)
    	end = time.time()
    	cputime = end -start
    	start = time.time()
    	gp.gpu_predict(testing)
    	end =time.time()
    	gputime = end - start
    	print N, cputime, gputime, cputime/gputime
    	#write testing set
   
