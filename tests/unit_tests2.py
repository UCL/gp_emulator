import numpy as np
#import matplotlib.pyplot as plt
from gp_emulator import MultivariateEmulator, GaussianProcess
from types import MethodType
import scipy.spatial.distance as dist
import os


#d = np.loadtxt("../data/test_00100.txt", delimiter=",")
d = np.loadtxt("../data/validation_set_001000.txt",delimiter=",")
gp = MultivariateEmulator(dump="../data/prosail_30_0_30_0.npz")


wv = np.arange (400, 2501)
#for isample in [0, 10, 20, 50]:
#    plt.plot ( wv, d[isample, 10:], '-', lw=2)
#    r = gp.predict(d[isample,:10])[0]
#    plt.plot ( wv, d[isample, 10:], '--', lw=2)



def get_testing_val (self, testing, prec, do_unc = True):  
    """Produce data for testing. It is a duplication of GaussianProcess.predict(), 
    for writting testing data. 
    """
    ( nn, D ) = testing.shape
    assert D == self.D
    expX=np.exp(self.theta)

    if prec == "float32":
        testing = np.float32(testing)
        self.inputs = np.float32(self.inputs)
        self.invQ = np.float32(self.invQ)
        self.invQt = np.float32(self.invQt)
        expX = np.float32(expX)


    
    t_expX = expX
    t_expXsqrt = np.sqrt(expX[:(self.D)])
    t_inputs = self.inputs
    t_testing = testing
    t_invQ = self.invQ
    
    cdist_test_var_1 = np.sqrt(expX[:(self.D)])*self.inputs
    cdist_test_var_2 = np.sqrt(expX[:(self.D)])*testing
    cdist_test_var_3 = dist.cdist ( np.sqrt(expX[:(self.D)])*self.inputs, np.sqrt(expX[:(self.D)])*testing, 'sqeuclidean')
    # avoid cdist function converting automatically the result to float64
    #if prec is 'float32':
    #    cdist_test_var_3 = np.float32(cdist_test_var_3)

    cdist_test_var_4 = expX[self.D]*np.exp(-0.5*cdist_test_var_3)
    mu = np.dot( cdist_test_var_4.T, self.invQt)
    
    var_test_1 = np.dot(self.invQ, cdist_test_var_4)     
    
    if prec is 'float32':
        assert cdist_test_var_1.dtype is np.dtype('float32')
        assert cdist_test_var_2.dtype is np.dtype('float32')
        assert cdist_test_var_3.dtype is np.dtype('float32')
        assert cdist_test_var_4.dtype is np.dtype('float32')
        
        assert var_test_1.dtype is np.dtype('float32')

    np.float64(t_expX).tofile("./data/expX.bin")
    np.float64(t_expXsqrt).tofile("./data/expXsqrt.bin")
    np.float64(t_inputs).tofile("./data/inputs.bin")
    np.float64(t_testing).tofile("./data/testing.bin")
    np.float64(cdist_test_var_1).tofile("./data/cdist_test_var1.bin")
    np.float64(cdist_test_var_2).tofile("./data/cdist_test_var2.bin")
    np.float64(cdist_test_var_3).tofile("./data/cdist_a.bin")
    np.float64(cdist_test_var_4).tofile("./data/cdist_expa.bin")    
        
    np.float64(t_invQ).tofile("./data/invQ.bin")
    np.float64(mu).tofile("./data/mu.bin")
    np.float64(var_test_1).tofile("./data/var_test1.bin")
    np.float64(self.invQt).tofile("./data/invQt.bin")


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

    np.float64(var).tofile("./data/var.bin")
    np.float64(deriv).tofile("./data/deriv.bin")

    if do_unc:
        return mu, var, deriv
    else:
        return mu, deriv


def perband_emulators ( emulators, band_pass ):
    """This function creates per band emulators from the full-spectrum
    emulator. Should be faster in many cases"""
    
    n_bands = band_pass.shape[0]
    x_train_pband = [ emulators.X_train[:,band_pass[i,:]].mean(axis=1) \
        for i in xrange( n_bands ) ]
    x_train_pband = np.array ( x_train_pband )
    emus = []
    # add get_testing_val 
    GaussianProcess.get_testing_val = MethodType(get_testing_val, None, GaussianProcess)

    for i in xrange( n_bands ):
        gp = GaussianProcess ( emulators.y_train[:]*1, \
                x_train_pband[i,:] )
        print 'p1', (emulators.y_train[:]*1).shape, 'p2', x_train_pband[i,:].shape
        gp.learn_hyperparameters ( n_tries=5 )
        emus.append ( gp )
    return emus


precision = "float64"
new_output = d[:,10:][:,452:486].mean(axis=1)
band_pass = np.zeros( (1,2101), dtype=np.bool)
band_pass[:, 442:476] = 1
gp_single = perband_emulators ( gp, band_pass )


print isinstance ( gp_single[0], GaussianProcess)
X = d[:, :10]
gpur = gp_single[0].predict ( X ,is_gpu=True, prec = precision)
cpur = gp_single[0].predict ( X, is_gpu=False)
print max(abs(gpur[0]-cpur[0]))

gp_single[0].get_testing_val(X, prec = 'float64')
command = "./gpu_predict_test  %d %d %d" % (250, 1000, 10)
os.system(command)

for i in xrange(3):
    val_absolute = np.mean(np.abs(cpur[i]))
    error_absolute = np.abs( cpur[i] - gpur[i] )
    error_relative = error_absolute / np.abs( cpur[i] )
    print '\n--------------'
    print 'output', i, precision 
    print 'absolute_val\t%.2g'%val_absolute
    print 'error_absolute\t', '%.2g [average]\t%.2g [max]'% (np.mean(error_absolute), np.max(error_absolute))
    print 'error_relative\t', '%.2g [average]\t%.2g [max]'% (np.mean(error_relative), np.max(error_relative))





