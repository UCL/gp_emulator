#!/usr/local/bin/python

import numpy as np
import scipy.spatial.distance as dist
import _gpu_predict
import time
from gp_emulator import GaussianProcess 
from types import MethodType
import sys

def set_testing_val (self, Ninputs, Npredict, Ntrain):
    self.D = Ninputs
    self.theta = np.random.random((Ninputs+2))
    self.invQ = np.random.random(( Ntrain, Ntrain ))
    self.invQt = np.random.random ((Ntrain))


def predict_benchmark( self, testing, do_unc=True ):
    start_predict = time.time()
    ( nn, D ) = testing.shape
    assert D == self.D
    expX = np.exp ( self.theta )
    
    start_mvel = time.time()
    in_dist1 = np.sqrt(expX[:(self.D)])*self.inputs
    in_dist2 = np.sqrt(expX[:(self.D)])*testing
    end_mvel = time.time()
    print 'mvel',2, end_mvel - start_mvel

    start_dist = time.time()
    a = dist.cdist ( in_dist1, in_dist2, 'sqeuclidean')
    end_dist = time.time()
    print 'dist',1, end_dist - start_dist

    start_exp = time.time()
    a = expX[self.D]*np.exp(-0.5*a)
    end_exp = time.time()
    print 'exp-ele',3, end_exp - start_exp

    b = expX[self.D]
    
    start_mv =time.time()
    mu = np.dot( a.T, self.invQt)
    end_mv = time.time()
    print 'mv',1,end_mv - start_mv
    
    if do_unc:
        start_mm = time.time()
        k = np.dot(self.invQ,a)
        end_mm = time.time()
        print 'mm',1,end_mm - start_mm

        start_mulele = time.time()
        k = a * k 
        end_mulele = time.time()
        print 'mul-ele',1, end_mulele - start_mulele

        start_sum = time.time()
        var = b - np.sum (b, axis=0)
        end_sum = time.time()
        print 'sum',1, end_sum - start_sum
        # Derivative and partial derivatives of the function
    deriv = np.zeros ( ( nn, self.D ) )

    crossminus = 0
    mul_ele2 = 0
    mv2 = 0

    for d in xrange ( self.D ):
        s_crossminus = time.time()
        aa = self.inputs[:,d].flatten()[None,:] - testing[:,d].flatten()[:,None]
        e_crossminus = time.time()
        crossminus = crossminus + e_crossminus - s_crossminus

        s_mul_ele2 = time.time()
        c = a*aa.T
        e_mul_ele2 = time.time()
        mul_ele2 = mul_ele2 + e_mul_ele2 - s_mul_ele2

        s_mv2 = time.time()
        deriv[:, d] = expX[d]*np.dot(c.T, self.invQt)
        e_mv2 = time.time()
        mv2 = mv2 + e_mv2 - s_mv2
    end_predict = time.time()
       
    print 'crossminus',10, crossminus
    print 'mul_ele2',10,mul_ele2
    print 'mv2',10, mv2
    print 'predict',1,end_predict - start_predict

    if do_unc:
        return mu, var, deriv
    else:
        return mu, deriv


if __name__ == '__main__':

    GaussianProcess.set_testing_val = MethodType(set_testing_val, None, GaussianProcess)
    GaussianProcess.predict_benchmark = MethodType(predict_benchmark, None, GaussianProcess)
    print 'Problem_size\tCPU time\tGPU time\tSpeedup\tStatus'
    print '-----------------------------'
    
    for i in xrange(3):
        
        Npredict = np.int(1e6)
	Ntrain = 250
        Ninputs = 10

        inputs = np.random.random(( Ntrain, Ninputs))
        testing = np.random.random(( Npredict, Ninputs))
        
        gp = GaussianProcess(inputs, [])
        gp.set_testing_val(Ninputs, Npredict, Ntrain)

        #CPU predict
    	start = time.time()
    	[mu_c, var_c, deriv_c] = gp.predict_benchmark(testing)
    	end = time.time()
    	cputime = end -start
	print 'cputime', cputime
	print '=================='


        #GPU PREDICT
        start = time.time()
        [mu_g, var_g, deriv_g] = gp.pure_c_predict(testing, precision=np.float32)
        end = time.time()
        gputime = end - start
        print 'cputime', gputime
        print '=================='

