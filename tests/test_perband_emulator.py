import numpy as np
import matplotlib.pyplot as plt
from gp_emulator import MultivariateEmulator, GaussianProcess



#d = np.loadtxt("test_00100.txt", delimiter=",")
d = np.loadtxt("data/validation_set_001000.txt",delimiter=",")
gp = MultivariateEmulator(dump="data/prosail_30_0_30_0.npz")


wv = np.arange (400, 2501)
#for isample in [0, 10, 20, 50]:
#    plt.plot ( wv, d[isample, 10:], '-', lw=2)
#    r = gp.predict(d[isample,:10])[0]
#    plt.plot ( wv, d[isample, 10:], '--', lw=2)


r = gp.predict(d[0,:10],is_gpu=True)[0]



def perband_emulators ( emulators, band_pass ):
    """This function creates per band emulators from the full-spectrum
    emulator. Should be faster in many cases"""
    
    n_bands = band_pass.shape[0]
    x_train_pband = [ emulators.X_train[:,band_pass[i,:]].mean(axis=1) \
        for i in xrange( n_bands ) ]
    x_train_pband = np.array ( x_train_pband )
    emus = []
    for i in xrange( n_bands ):
        gp = GaussianProcess ( emulators.y_train[:]*1, \
                x_train_pband[i,:] )
        gp.learn_hyperparameters ( n_tries=5 )
        emus.append ( gp )
    return emus



#new_output = d[:,10:][:,452:486].mean(axis=1)
#band_pass = np.zeros( (1,2101), dtype=np.bool)
#band_pass[:, 442:476] = 1
#gp_single = perband_emulators ( gp, band_pass )


#print isinstance ( gp_single[0], GaussianProcess)
#X = d[:, :10]
#r = gp_single[0].predict ( X )[0]

#gp_single[0].gpu_predict ( X )
#print 'predict finish'
#plt.plot ( new_output, r,'o')
#plt.plot([0, 0.6], [0, 0.6], 'k--')


#%timeit r = gp_single[0].predict ( X )[0]
