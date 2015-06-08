import numpy as np
import matplotlib.pyplot as plt
from gp_emulator import MultivariateEmulator, GaussianProcess



#d = np.loadtxt("test_00100.txt", delimiter=",")
d = np.loadtxt("../data/validation_set_001000.txt",delimiter=",")
gp = MultivariateEmulator(dump="../data/prosail_30_0_30_0.npz")


wv = np.arange (400, 2501)
for isample in [0, 10, 20, 50]:
   print isample, "----------------"
#   plt.plot ( wv, d[isample, 10:], '-', lw=2)
   r = gp.predict(d[0:2,:10])[0]
#   plt.plot ( wv, d[isample, 10:], '--', lw=2)
#plt.show()

#r = gp.predict(d[[1,2,3,4,5],:10], is_gpu = True)[0]

#N = 1e3
#M = 250
#D = 10

#input_t = np.random.random((N, D))
#input_v = np.random.random((M, D))

#yields = np.random.random(N)
#gp = GaussianProcess(input_t, yields)
#gp.learn_hyperparameters (n_tries=1)
#pred_mu, pred_var, par_dev = gp.predict ( input_v, is_gpu=True)
#mu, var, dev = gp.predict ( input_v)


