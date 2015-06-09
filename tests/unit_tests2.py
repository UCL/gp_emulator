import numpy as np
import matplotlib.pyplot as plt
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
    
    t_expX.tofile("./data/expX.bin")
    t_expXsqrt.tofile("./data/expXsqrt.bin")
    t_inputs.tofile("./data/inputs.bin")
    t_testing.tofile("./data/testing.bin")
    cdist_test_var_1.tofile("./data/cdist_test_var1.bin")
    cdist_test_var_2.tofile("./data/cdist_test_var2.bin")
    cdist_test_var_3.tofile("./data/cdist_a.bin")
    cdist_test_var_4.tofile("./data/cdist_expa.bin")    
        
    t_invQ.tofile("./data/invQ.bin")
    mu.tofile("./data/mu.bin")
    var_test_1.tofile("./data/var_test1.bin")
    self.invQt.tofile("./data/invQt.bin")


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

    var.tofile("./data/var.bin")
    deriv.tofile("./data/deriv.bin")

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
    GaussianProcess.get_testing_val = MethodType(get_testing_val, None, GaussianProcess)

    for i in xrange( n_bands ):
        gp = GaussianProcess ( emulators.y_train[:]*1, \
                x_train_pband[i,:] )
        print 'p1', (emulators.y_train[:]*1).shape, 'p2', x_train_pband[i,:].shape
        gp.learn_hyperparameters ( n_tries=5 )
        emus.append ( gp )
    return emus



new_output = d[:,10:][:,452:486].mean(axis=1)
band_pass = np.zeros( (1,2101), dtype=np.bool)
band_pass[:, 442:476] = 1
gp_single = perband_emulators ( gp, band_pass )


print isinstance ( gp_single[0], GaussianProcess)
X = d[:, :10]
gpur = gp_single[0].predict ( X ,is_gpu=True)
cpur = gp_single[0].predict ( X, is_gpu=False)



gp_single[0].get_testing_val(X)
command = "./gpu_predict_test  %d %d %d" % (250, 1000, 10)
os.system(command)

#print X
print 'predict finish'
#plt.plot ( new_output, r,'o')
#plt.plot([0, 0.6], [0, 0.6], 'k--')

#plt.show()
#%timeit r = gp_single[0].predict ( X )[0]



