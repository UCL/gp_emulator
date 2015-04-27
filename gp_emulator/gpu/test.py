import numpy as np
import scipy.spatial.distance as dist
class GP:
    def __init__ (self, P, N, M):
        self.D = P
        self.theta = np.random.random((P+2))
        self.inputs = np.random.random ((M,P))
        self.invQ = np.random.random((M,M))
        self.invQt = np.random.random ((M))

    @profile
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


if __name__ == '__main__':
    N=4e5
    M=250
    P=10
    testing=np.ones((N,P))
    gp=GP(P,N,M)
    gp.predict(testing)
    
