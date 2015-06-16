import numpy as np
#import matplotlib.pyplot as plt
from gp_emulator import MultivariateEmulator, GaussianProcess
from types import MethodType
import scipy.spatial.distance as dist
import os


#d = np.loadtxt("../data/test_00100.txt", delimiter=",")
d = np.loadtxt("../data/validation_set_100000.txt",delimiter=",")
gp = MultivariateEmulator(dump="../data/prosail_vza30_sza0_saa0_vaa0_n250.npz")


wv = np.arange (400, 2501)
#for isample in [0, 10, 20, 50]:
#    plt.plot ( wv, d[isample, 10:], '-', lw=2)
#    r = gp.predict(d[isample,:10])[0]
#    plt.plot ( wv, d[isample, 10:], '--', lw=2)

def perband_emulators ( emulators, band_pass ):
    """This function creates per band emulators from the full-spectrum
    emulator. Should be faster in many cases"""
    
    n_bands = band_pass.shape[0]
    x_train_pband = [ emulators.X_train[:,band_pass[i,:]].mean(axis=1) \
        for i in xrange( n_bands ) ]
    x_train_pband = np.array ( x_train_pband )
    emus = []
    # add get_testing_val 
    #GaussianProcess.get_testing_val = MethodType(get_testing_val, None, GaussianProcess)

    for i in xrange( n_bands ):
        gp = GaussianProcess ( emulators.y_train[:]*1, \
                x_train_pband[i,:] )
        print 'p1', (emulators.y_train[:]*1).shape, 'p2', x_train_pband[i,:].shape
        gp.learn_hyperparameters ( n_tries=5 )
        emus.append ( gp )
    return emus


precision = "float32"
new_output = d[:,10:][:,452:486].mean(axis=1)
band_pass = np.zeros( (1,2101), dtype=np.bool)
band_pass[:, 442:476] = 1
gp_single = perband_emulators ( gp, band_pass )


print isinstance ( gp_single[0], GaussianProcess)
X = d[:, :10]
gpur = gp_single[0].predict ( X ,is_gpu=True, prec = precision)
cpur = gp_single[0].predict ( X, is_gpu=False)


for i in xrange(3):
    val_absolute = np.mean(np.abs(cpur[i]))
    error_absolute = np.abs( cpur[i] - gpur[i] )
    error_relative = error_absolute / np.abs( cpur[i] )
    print '\n--------------'
    print 'output', i, precision 
    print 'absolute_val\t%.2g'%val_absolute
    print 'error_absolute\t', '%.2g [average]\t%.2g [max]'% (np.mean(error_absolute), np.max(error_absolute))
    print 'error_relative\t', '%.2g [average]\t%.2g [max]'% (np.mean(error_relative), np.max(error_relative))





