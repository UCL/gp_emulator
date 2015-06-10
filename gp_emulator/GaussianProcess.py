# -*- coding: utf-8 -*-
import warnings
import numpy as np
import scipy.spatial.distance as dist
import random
import sys
#import _gpu_predict

def k_fold_cross_validation(X, K, randomise = False):
    """
    Generates K (training, validation) pairs from the items in X.

    Each pair is a partition of X, where validation is an iterable
    of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

    If randomise is true, a copy of X is shuffled before partitioning,
    otherwise its order is preserved in training and validation.
    ## {{{ http://code.activestate.com/recipes/521906/ (r3)
    """
    if randomise: 
        X = list(X)
        random.shuffle(X)
    for k in xrange(K):
        training = [x for i, x in enumerate(X) if i % K != k]
        validation = [x for i, x in enumerate(X) if i % K == k]
        yield training, validation

class GaussianProcess:
    """
    A simple class for Gaussian Process emulation. Currently, it assumes
    a squared exponential covariance function, but other covariance
    functions ought to be possible and easy to implement.
    
    """
    def __init__ ( self, inputs, targets ):
        """The inputs are the input vectors, whereas the targets are the
        emulated model outputs.

	Parameters
	-----------
	inputs: array size (Ntrain x Ninputs)
		An input array of size Ntrain * Ninputs (Ntrain is the 
		number of training samples, Ninputs is the dimensionality
                 of the input vector)
        targets: array size Ntrain
                The model outputs corresponding to the ``inputs`` training set
        """
		
        self.inputs = inputs
        self.targets = targets
        ( self.n, self.D ) = self.inputs.shape
    def _prepare_likelihood ( self ):
        """
        This method precalculates matrices and stuff required for the i
	log-likelihood maximisation routine, so that they can be
        reused when calling the ``predict`` method repeatedly.
        """
        
        # Exponentiate the hyperparameters
        
        exp_theta = np.exp ( self.theta )
        # Calculation of the covariance matrix Q using theta
        self.Z = np.zeros ( (self.n, self.n) )
        for d in xrange( self.D ):
            self.Z = self.Z + exp_theta[d]*\
                            ((np.tile( self.inputs[:, d], (self.n, 1)) - \
                              np.tile( self.inputs[:, d], (self.n, 1)).T))**2
        self.Z = exp_theta[self.D]*np.exp ( -0.5*self.Z)
        self.Q = self.Z +\
            exp_theta[self.D+1]*np.eye ( self.n )
        self.invQ = np.linalg.inv ( self.Q )
        self.invQt = np.dot ( self.invQ, self.targets )

        self.logdetQ = 2.0 * np.sum ( np.log ( np.diag ( \
                        np.linalg.cholesky ( self.Q ))))


    def loglikelihood ( self, theta ):
        """Calculates the loglikelihood for a set of hyperparameters
        ``theta``. The size of ``theta`` is given by the dimensions of
	the input vector to the model to be emulated.

	Parameters
	----------
	theta: array
		Hyperparameters
	"""
        self._set_params ( theta )
        
        loglikelihood = 0.5*self.logdetQ + \
                        0.5*np.dot ( self.targets, self.invQt ) + \
                        0.5*self.n*np.log(2.*np.pi )
        self.current_theta = theta
        self.current_loglikelihood = loglikelihood
        return loglikelihood

    def partial_devs ( self, theta ):
	"""This function calculates the partial derivatives of the 
	cost function as a function of the hyperameters, and is only
	needed during GP training.

	Parameters
	-----------
	theta: array
		Hyperparameter set
	"""
        partial_d = np.zeros ( self.D + 2 )
        
        for d in xrange ( self.D ):
            V = ((( np.tile ( self.inputs[:, d], ( self.n, 1)) - \
                np.tile ( self.inputs[:, d], ( self.n, 1)).T))**2).T*self.Z
            
            partial_d [d] = np.exp( self.theta[d] )* \
                             ( np.dot ( self.invQt, np.dot ( V, self.invQt) ) - \
                              np.sum ( self.invQ*V))/4.
            
            
        partial_d[ self.D ] = 0.5*np.sum ( self.invQ*self.Z ) - \
                              0.5*np.dot ( self.invQt, \
                                        np.dot ( self.Z, self.invQt))
        partial_d [ self.D + 1 ] = 0.5*np.trace( self.invQ )*\
                        np.exp( self.theta[self.D+1] ) - \
                        0.5*np.dot (self.invQt, self.invQt ) * \
                        np.exp( self.theta[self.D + 1])
        return partial_d
        
    def _set_params ( self, theta ):
	"""Sets the hyperparameters, and thus also precalculates terms
	that depend on them. Since hyperparameters are fixed after
	training, this speeds up some calculations.
	
	Parameters
	-----------
	theta: array
		hyperparameters
`	"""
        
        self.theta = theta
        self._prepare_likelihood ( )
        
    def _learn ( self, theta0, verbose ):
	"""The training method, called ''learn'' to keep up with the
	trendy Machine Learning kids!
	Takes an initial guess of the hyperparameters, and minimises 
	that through a gradient descent algorithm, using methods
	``likelihood`` and ``partial_devs`` to select hyperparameters
	that result in a minimal log-likelihood.

	Parameters
	-----------
	theta0: array
		Hyperparameters
	verbose: flag
		Whether to provide lots of information on the 
		minimiation. Useful to see whether its fitting or
		not for some hairy problems.
	"""
        # minimise self.loglikelihood (with self.partial_devs) to learn
        # theta
        from scipy.optimize import fmin_cg,fmin_l_bfgs_b
        self._set_params ( theta0 )
        if verbose:
            iprint = 1
        else:
            iprint = -1
        try:
            #theta_opt = fmin_cg ( self.loglikelihood,
            #        theta0, fprime = self.partial_devs, \
            #        full_output=True, \
            #        retall = 1, disp=1 )
            theta_opt = fmin_l_bfgs_b(  self.loglikelihood, \
                     theta0, fprime = self.partial_devs, \
                     factr=0.1, pgtol=1e-20, iprint=iprint)
        except np.linalg.LinAlgError:
            warnings.warn ("Optimisation resulted in linear algebra error. " + \
                "Returning last loglikelihood calculated, but this is fishy", \
                    RuntimeWarning )
            #theta_opt = [ self.current_theta, self.current_loglikelihood ]
            theta_opt = [ self.current_theta, 9999]
            
        return theta_opt

    def learn_hyperparameters ( self, n_tries=15, verbose=False ):
	"""User method to fit the hyperparameters of the model, using
	random initialisations of parameters. The user should provide
	a number of tries (e.g. how many random starting points to
	avoid local minima), and whether it wants lots of information
	to be reported back.
	
	Parameters
	-----------
	n_tries: int, optional
		Number of random starting points
	verbose: flag, optional
		How much information to parrot (e.g. convergence of
		the minimisation algorithm)

	"""
        log_like = []
        params = []
        for theta in 5.*(np.random.rand(n_tries, self.D+2) - 0.5):
            T = self._learn ( theta ,verbose )
            log_like.append ( T[1] )
            params.append ( T[0] )
        log_like = np.array ( log_like )
        idx = np.argsort( log_like )[0]
        print "After %d, the minimum cost was %e" % ( n_tries, log_like[idx] )
        self._set_params ( params[idx])
        return (log_like[idx], params[idx] )

    def cpu_predict ( self, testing, do_unc=True ):
	"""Make a prediction for a set of input vectors, as well as 
	calculate the partial derivatives of the emulated model, 
	and optionally, the "emulation uncertainty". 

	Parameters
	-----------
	testing: array, size Npred * Ninputs
		The size of this array (and it must always be a 2D array!)
		is given by the number of input vectors that will be run
		through the emulator times the input vector size.

	do_unc: flag, optional
		Calculate the uncertainty (if you don't set this flag, it
		can shave a few us"""
       

        ( nn, D ) = testing.shape
        assert D == self.D
        expX = np.exp ( self.theta )
        
        a = dist.cdist ( np.sqrt(expX[:(self.D)])*self.inputs, \
            np.sqrt(expX[:(self.D)])*testing, 'sqeuclidean')
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
        
        
    def gpu_predict ( self, testing, prec, threashold):# self, testing, do_unc=True):
        import _gpu_predict
        ( nn, D ) = testing.shape
        assert D == self.D
        expX = np.exp ( self.theta )
        
        N=testing.shape[0]
        M=self.inputs.shape[0]
        theta_size=self.theta.size

        nblocks = np.int(np.ceil( np.float32( nn ) / threashold ))
        block_size = np.int(np.ceil( nn / nblocks ))
        ind_start = range( np.int(0), np.int(nn), np.int(block_size) )
        ind_end = np.append( ind_start[1:], nn )
        nblocks = len(ind_start)
        
        if nblocks > 1:
            last_two_block_size = ( ind_end[ nblocks - 1 ] - ind_start[ nblocks - 2 ] ) / 2
            ind_end[ nblocks - 2 ] = ind_start[ nblocks - 2 ] + last_two_block_size
            ind_start[ nblocks - 1 ] = ind_end[ nblocks - 2 ]
        print ind_end-ind_start
        assert np.all(ind_end - ind_start > 500 )
        assert np.all(ind_end - ind_start < threashold * 1.5 )
            


        #stretch matrix to vector to feed predict_wrap()
        inputs = self.inputs.reshape(self.inputs.shape[0] * self.inputs.shape[1])
        invQt = self.invQt
        invQ =  self.invQ.reshape(self.invQ.shape[0] * self.invQ.shape[1])

        if prec == "float32":
            inputs = np.float32( inputs )
            invQt = np.float32( invQt )
            invQ = np.float32( invQ )
            expX = np.float32( expX )

        mu = []
        var = []
        deriv = np.array([]).reshape( ( 0, D) )

        for block_id in xrange(len(ind_start)):
           testing_block = testing[ind_start[block_id]:ind_end[block_id],:]   
           testing_block = testing_block.reshape( testing_block.shape[0] * testing_block.shape[1] )

           N_this_block = np.int(ind_end[block_id] - ind_start[block_id])
           mu_block = np.zeros( N_this_block )
           var_block = np.zeros( N_this_block )
           deriv_block = np.zeros( N_this_block * D )

           if prec == "float32":
               testing_block = np.float32(testing_block)
               mu_block = np.float32(mu_block)
               var_block = np.float32(var_block)
               deriv_block = np.float32(deriv_block)
 
           _gpu_predict.predict_wrap(
                   expX, inputs, invQt, invQ, testing_block,
                   mu_block, var_block, deriv_block,
                   N_this_block, M, D, theta_size)

           mu = np.append(mu, mu_block)
           var = np.append(var, var_block)
           deriv = np.append( deriv, deriv_block.reshape( D, N_this_block ).T, axis = 0)

        return mu, var, deriv



    def predict(self, testing, do_unc = True, is_gpu = False, prec = 'double', threashold = 2e5):
        ( nn, D ) = testing.shape
        if is_gpu == True:
            return self.gpu_predict(testing, prec, threashold = threashold)
        else:
            return self.cpu_predict(testing, do_unc)


        
    def hessian ( self, testing ):
        '''calculates the hessian of the GP for the testing sample. 
           hessian returns a (nn by d by d) array
        '''
        ( nn, D ) = testing.shape
        assert D == self.D
        expX = np.exp ( self.theta )
        aprime = dist.cdist ( np.sqrt(expX[:(self.D)])*self.inputs, \
                np.sqrt(expX[:(self.D)])*testing, 'sqeuclidean')
        a = expX[self.D]*np.exp(-0.5*aprime)
        dd_addition = np.identity(self.D)*expX[:(self.D)]
        hess = np.zeros ( ( nn, self.D , self.D) )
        for d in xrange ( self.D ):
            for d2 in xrange(self.D):
                aa = expX[d]*( self.inputs[:,d].flatten()[None,:] - 
                               testing[:,d].flatten()[:,None] )*   \
                     expX[d2]*( self.inputs[:,d2].flatten()[None,:] - 
                                testing[:,d2].flatten()[:,None] ) -  \
                     dd_addition[d,d2]
                cc = a*(aa.T)
                hess[:, d,d2] = np.dot(cc.T, self.invQt)
        return hess



        


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.set_printoptions(precision=2, suppress=True)
    #input_obs = np.array ( [[-4, -3, -1, 0, 2]]).T
    #target1 = np.array ([-2, 0, 1., 2., -1])
    #data = np.loadtxt ("mvreg.dat", delimiter="," )
    #target1 = data[ :,0]
    #target2 = data[ :,1]
    #target3 = data[ :,2]
    #input_obs = data[ :, 3: ]
    wheat = np.loadtxt("../data/argentine_wheat.dat")[:, 1:]
    yields = wheat[:,0]
    mu = yields.mean()
    sigma = yields.std()
    yields = (yields - mu ) / sigma
    wheat [:, 0] = yields
    rmse = []
    print wheat.shape
    for ( train,validate) in k_fold_cross_validation ( wheat, 5, randomise=True):
        train = np.array ( train )
        validate = np.array ( validate )
        yields_t = train [ :, 0]
        inputs_t = train [ :, 1:]
        yields_v = validate [ :,  0]
        inputs_v = validate [ :, 1:]
        #print inputs_t.shape, yields_t.shape, inputs_v.shape
        gp = GaussianProcess ( inputs_t, yields_t )
        theta_min= gp.learn_hyperparameters (n_tries=2)
        pred_mu, pred_var, par_dev = gp.predict ( inputs_v )
        #gp.gpu_predict ( inputs_v )
        #print "TEST"
        #print inputs_v
        #print pred_mu, pred_var, par_dev
        #r = ( yields_v - pred_mu )**2#/pred_var
        #rmse.append ( [ np.sqrt(r.mean()), theta_min[1] ])
        
        
    #gp.theta[2] = 0.
    ######theta = gp.theta
    ######print theta
    ######gp.predict ( input_obs )
    
    ######x = np.linspace(-5,5,100)
    
    ######(mu, var ) = gp.predict ( x[:, np.newaxis])
    ######plt.plot ( x, mu, '-r' )
    ######plt.fill_between ( x, mu+np.sqrt(var), mu-np.sqrt(var), color='0.8')
    #######plt.errorbar ( x, mu, yerr=np.sqrt(var)*0.5 )
    ######plt.plot ( input_obs, target1, 'gs' )
    ######plt.title("theta: %s" % theta )
    ######plt.show()



