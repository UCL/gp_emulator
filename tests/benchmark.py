#!/usr/local/bin/python
import numpy as np
import scipy.spatial.distance as dist
import _gpu_predict
import time
from gp_emulator import GaussianProcess 
from types import MethodType

def set_testing_val (self, P, N, M):
    self.D = P
    self.theta = np.random.random((P+2))
    self.invQ = np.random.random((M,M))
    self.invQt = np.random.random ((M))


if __name__ == '__main__':

    GaussianProcess.set_testing_val = MethodType(set_testing_val, None, GaussianProcess)
    print 'Problem_size\tCPU time\tGPU time\tSpeedup\tStatus'
    print '-----------------------------'

    for i in xrange(np.int(1e3), np.int(1e7), np.int(1e3)):
        N=i
        M=250
        P=10

        inputs = np.random.random((M,P))
        testing=np.random.random((N,P))
        
        gp = GaussianProcess(inputs, [])
        gp.set_testing_val(P, N, M)

        #CPU predict
    	start = time.time()
    	[mu_c, var_c, deriv_c] = gp.predict(testing, is_gpu=False )
    	end = time.time()
    	cputime = end -start
    	
        #GPU predict
        start = time.time()
    	[mu_g, var_g, deriv_g] = gp.predict(testing, is_gpu = True, prec = 'float32', threashold = 1e3)
    	end =time.time()
    	gputime = end - start
        print "%d\t%.2fs\t%.2fs\t%.2fx\t" % (N, cputime, gputime, cputime/gputime),

    
    
        # checking results
        try: 
            e_mu  = max(abs(mu_c - mu_g)) / np.max(abs(mu_c))
            e_var = max(abs(var_c - var_g)) / np.max(abs(var_c))
            e_deriv = np.max(abs(deriv_c - deriv_g)) / np.max(abs(deriv_c))
        except ValueError:
            print 'Results have invalid data type or dimension.'
        if e_mu > 1e-5 or e_var > 1e-5 or e_deriv > 1e-5:
            print 'Failed: ', 'e_mu=', e_mu, 'e_var=', e_var, 'e_deriv=',e_deriv
#            break
        else:
            print 'Pass'

