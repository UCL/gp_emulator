# -*- coding: utf-8 -*-
import warnings
import numpy as np
import scipy.spatial.distance as dist
import random
import sys
import _gpu_predict

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

    def get_gpu_block( self, size, block_size ):
        '''
        Distribute a size long vector block_size, and return the start index
        and end index of each block. To ensure size of the last block 
        is not too small, the last two blocks will have a equal size.  
        '''
        ind_start = range( np.int(0), np.int(size), np.int(block_size) )
        ind_end = np.append( ind_start[1:], size )
        nblocks = len(ind_start)
        
        # compute the size of last two blocks.
        if nblocks > 1:
            last_two_block_size = ( ind_end[ nblocks - 1 ] - ind_start[ nblocks - 2 ] ) / 2
            ind_end[ nblocks - 2 ] = ind_start[ nblocks - 2 ] + last_two_block_size
            ind_start[ nblocks - 1 ] = ind_end[ nblocks - 2 ]
       
        assert np.all(ind_end - ind_start <= block_size )
        return ind_start, ind_end

        
    def gpu_predict ( self, testing, precision, threshold):
        '''
        Parameters:
        --------------
        testing: 2D array n_predict * n_inputs
        precision: np.float32 / np.float64
        threshold: see predict() threshold.
        '''
        import _gpu_predict
        n_predict, n_inputs = testing.shape
        n_train = self.inputs.shape[0]
        theta_size=self.theta.size
        
        assert n_inputs == self.D
           
        #_predict_wrap() has to be fed by one dimentional array
        inputs = precision(self.inputs.reshape(self.inputs.shape[0] * self.inputs.shape[1]))
        invQt = precision(self.invQt)
        invQ =  precision(self.invQ.reshape(self.invQ.shape[0] * self.invQ.shape[1]))
        expX = precision(np.exp(self.theta))

        result = []
        error = []
        deriv = np.array([]).reshape((0, n_inputs))
        ind_start, ind_end = self.get_gpu_block( n_predict, threshold )

        for block_start, block_end in zip(ind_start, ind_end):
           testing_block = testing[block_start:block_end,:]   
           testing_block = testing_block.reshape( testing_block.shape[0] * testing_block.shape[1] )

           n_predict_block = np.int(block_end - block_start)
           result_block = np.zeros( n_predict_block )
           error_block = np.zeros( n_predict_block )
           deriv_block = np.zeros( n_predict_block * n_inputs )
           
           testing_block = precision(testing_block)
           result_block = precision(result_block)
           error_block = precision(error_block)
           deriv_block = precision(deriv_block)
 
           _gpu_predict.predict_wrap(
                   expX, inputs, invQt, invQ, testing_block,
                   result_block, error_block, deriv_block,
                   n_predict_block, n_train, n_inputs, theta_size)

           result = np.append(result, result_block)
           error = np.append(error, error_block)
           #deriv produced by gpu is transposed, so here we need transpose them back.
           deriv = np.append(deriv, deriv_block.reshape( n_inputs, n_predict_block ).T, axis = 0)
        return result, error, deriv

    def pure_c_predict(self, testing, precision):
        import _gpu_predict
        n_predict, n_inputs = testing.shape
        n_train = self.inputs.shape[0]
        theta_size=self.theta.size

        assert n_inputs == self.D

        #_predict_wrap() has to be fed by one dimentional array
        testing = precision(testing.reshape(testing.shape[0]*testing.shape[1]))
        inputs = precision(self.inputs.reshape(self.inputs.shape[0] * self.inputs.shape[1]))
        invQt = precision(self.invQt)
        invQ =  precision(self.invQ.reshape(self.invQ.shape[0] * self.invQ.shape[1]))
        expX = precision(np.exp(self.theta))

        result = precision(np.zeros(n_predict))
        error = precision(np.zeros(n_predict))
        deriv = precision(np.zeros(n_predict * n_inputs))
        
        _gpu_predict.pure_c_predict_wrap(
                   expX, inputs, invQt, invQ, testing,
                   result, error, deriv,
                   n_predict, n_train, n_inputs, theta_size)
        return result, error, deriv.reshape(n_inputs, n_predict).T


    def predict(self, testing, do_unc = True, is_gpu = False, precision = np.float64, threshold = 2e5):
        '''
        Parameters:
        --------------
        testing: 2D array n_predict * n_inputs
        precision: np.float32 / np.float64
        do_unc: tag to switch on or off calculation of the uncertainty. It can only affect cpu_predict()
        threshold: is the maximum number of n_predict that gpu_predict() can deal with.
                If the n_predict is larger than the threshold, data will be truncated,
                and gpu_predict will be excuted for multiple time.
        '''
        if is_gpu == True:
            return self.pure_c_predict(testing = testing, precision = precision)
            #return self.gpu_predict(testing, precision, threshold = threshold)
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
