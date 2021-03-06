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


if __name__ == '__main__':

    GaussianProcess.set_testing_val = MethodType(set_testing_val, None, GaussianProcess)
    print 'Problem_size\tCPU time\tGPU time\tSpeedup\tStatus'
    print '-----------------------------'
    
    for Npredict in xrange(np.int(1e5), np.int(1e6), np.int(1e5)):
        Ntrain = 250
        Ninputs = 10

        inputs = np.random.random(( Ntrain, Ninputs))
        testing = np.random.random(( Npredict, Ninputs))
        
        gp = GaussianProcess(inputs, [])
        gp.set_testing_val(Ninputs, Npredict, Ntrain)

        #CPU predict
    	start = time.time()
    	[mu_c, var_c, deriv_c] = gp.predict(testing, is_gpu=False )
    	end = time.time()
    	cputime = end -start
    	
        #GPU predict
        start = time.time()
        [mu_g, var_g, deriv_g] = gp.predict(testing, is_gpu = True, precision = np.float32, threshold = 1e5)
    	end =time.time()
    	gputime = end - start
        print "%d\t%.2fs\t%.2fs\t%.2fx\t" % (Npredict, cputime, gputime, cputime/gputime),

    
    
        # checking results
        try: 
            e_mu  = max(abs(mu_c - mu_g) ) / np.max(abs(mu_c))
            e_var = max(abs(var_c - var_g)) / np.max(abs(var_c))
            e_deriv = np.max(abs(deriv_c - deriv_g)) / np.max(abs(deriv_c))
        except ValueError:
            print 'Results have invalid data type or dimension.'
        if e_mu > 1e-5 or e_var > 1e-5 or e_deriv > 1e-5:
            print 'FAILED\t',
        else:
            print 'Pass\t',
        print 'e_mu=%.2g\te_var=%.2g\te_deriv=%.2g\t'%(e_mu, e_var, e_deriv)


