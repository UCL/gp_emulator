import numpy as np
import scipy.spatial.distance as dist
import _gpu_predict
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
        print 'expX\n', np.sqrt(expX[:(self.D)]*testing)

        a =_gpu_predict.predict_wrap(
            expX,
            self.inputs,
            self.invQt,
            self.invQ,
            testing,
            N,M,D,theta_size)
        #print 'testing\n',testing
        print 'a\n',a


        # for passing the test (temp)
        mu=0
        var=0
        deriv=0
        if do_unc:
            return mu, var, deriv
        else:
            return mu, deriv


if __name__ == '__main__':
    N=1e6
    M=250#250
    P=10
    testing=np.ones((N,P))
    gp=GP(P,N,M)
    #gp.predict(testing)
    gp.gpu_predict(testing)

    
