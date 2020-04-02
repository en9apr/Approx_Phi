#!/usr/bin/env python3
"""
============================================================================
Surrogate modelling methods based on Gaussian processes
============================================================================


:Author:
         Alma Rahat <a.a.m.rahat@exeter.ac.uk>
:Date:
         16 May, 2016 
:Modified:
         1 December 2016
:File:
         surrogate.py
"""

# imports
import numpy as np
import GPy as GP
from pyDOE import lhs as LHS
from scipy.special import erf as ERF
from scipy.spatial import distance


from deap import algorithms
from deap import base
from deap import benchmarks
from deap import cma
from deap import creator
from deap import tools


import math, warnings, time
import sys
import matplotlib.pyplot as plt
plt.ion()
np.set_printoptions(precision=8)

class Surrogate(object):
    """
    A surrogate implementation based on the GPy module. See the following 
    webpage for detailed documentation on GPy: https://github.com/SheffieldML/GPy
    """

    def __init__(self, init_X, init_Y, kernel, restarts=10, verbose=False):
        """The constructor creates a surrogate model based on supplied training 
        data.
        
        parameters:
        -----------        
        init_X: a numpy array of initial input parameters. This should be structured 
                such that init_X.shape = (n,), where, 
                n = number of sample points.
                And for each element x in fronts should be structured such that 
                x.shape = (1, D), where, 
                D = number of dimensions.
        init_Y: a numpy array of actual function evaluations (associated with init_X).
                The structure should be init_Y.shape = (n, 1), where n is the number
                of sample points.
        kernel: GPY kernels. See the following webpage for all avaialbel options.
                http://gpytest2.readthedocs.io/en/latest/GPy.kern.html
        restarts: Number of restarts for the optimisation of the hyperparameters. 
        verbose: whether to print debug information.
        """
        print('Creating surrogate...')
        self.X = init_X
        self.Y = init_Y
        self.kernel = kernel
        self.restarts = restarts
        self.verbose = verbose
        # Setup scaling and standardised data
        self.xbar = np.mean(self.X, axis=0)
        self.xstd = np.std(self.X, axis=0)
        self.Xtr = (self.X-self.xbar)/self.xstd
        self.ybar = np.mean(self.Y)
        self.ystd = np.std(self.Y)
        if self.ystd!=0:
            self.Ytr = (self.Y-self.ybar)/self.ystd
        else:
            self.Ytr = self.Y.copy() 
        self.model = self.train()
        
        # APR added
        self.pen_locations = None
        self.r_x0 = None
        self.s_x0 = None
        self.L = None
        self.best = None
        # APR added
        
        
        
    def train(self, fix_min=1e-4, fix_max=1e8, optimizer='lbfgs', \
                n_restarts=10, n_proc=1):
        """A method to train the GP.
        
        Parameters:
        -----------
        fix_min (float): minimum noise variance; zero measurement variance may 
                            lead to negative predictive variance. 
        fix_max (float): maximum limit for hyperparameters.
        optimiser (str): hyperparameter optimiser name. Consult GPy 
                            documentation for avaialble optimisers.
        n_restarts (int): number of restarts for hyper-parameter optimisation.
        n_proc (int): number of processors to use.
        """
        print('Optimiser name: ', optimizer)
        model = GP.models.GPRegression(self.Xtr, self.Ytr, self.kernel)
        model.constrain_positive('')
        pnames = model.parameter_names()
        model[pnames[0]].constrain_bounded(fix_min, fix_max)
        model[pnames[1]].constrain_bounded(fix_min, fix_max)
        model[pnames[-1]].constrain_fixed(fix_min)
        opt_runs = model.optimize_restarts(optimizer=optimizer, \
                                            num_restarts=n_restarts,\
                                            num_processes=n_proc, \
                                            verbose=self.verbose)
        # check models for correctness
        fopts = np.array([opt.f_opt for opt in opt_runs])
        check_nan = np.isnan(fopts)
        if np.all(check_nan):
            print("All hyper-parameter optimisation runs resulted"+\
                          "in NaN value."+\
                          "\nThis is likely becuase errors occured while"+\
                          " calculating the negative log likelihood."+\
                          "\nWe will try to run simplex algorithm instead.")
            model = GP.models.GPRegression(self.Xtr, self.Ytr, self.kernel)
            model.constrain_positive('')
            if fix_noise is not None:
                model['Gaussian_noise'].constrain_fixed(fix_noise)
            opt_runs = model.optimize_restarts(optimizer='simplex', \
                                                num_restarts=n_restarts,\
                                                num_processes=n_proc, \
                                                verbose=self.verbose)
            fopts = np.array([opt.f_opt for opt in opt_runs])
            check_nan = np.isnan(fopts)
            if np.all(check_nan == True):
                self.save_data('nan_debug', self.Xtr, self.Ytr)
                raise Exception('Hyper-parameter optimisation has failed.')
        if np.any(check_nan):
            inds = np.where(check_nan)[0]
            valid_runs = np.delete(np.arange(n_restarts, dtype='int'), inds)
            print("The following hyper-parameter optimisation runs"+\
                           " resulted in NaN value: " + str(inds) +\
                           "\nThis is likely becuase errors occured while"+\
                           " calculating the negative log likelihood."+\
                           "\nSetting hyper-parameters to the known best.")
            fopt_ind = np.argmin(fopts[valid_runs])
            model.optimizer_array = opt_runs[valid_runs[fopt_ind]].x_opt       
        return model

    def save_data(self, filename, xtr, ytr):
        '''
        Save traing data in CSV files. 
        
        Parameters.
        -----------
        filename (str): destination file name.
        xtr (np.array): training decision vectors. 
        ytr (np.array): training objective vector.
        '''
        f = open(filename+'_xtr.csv', 'wb')
        np.savetxt(f, xtr, delimiter=',')
        f.close()
        f = open(filename+'_ytr.csv', 'wb')
        np.savetxt(f, ytr, delimiter=',')
        f.close()

    #@profile
    def predict(self, x):
        """
        Predict the mean and the standard deviation for a given set of 
        decision vectors. 
        
        Parameters:
        -----------
        x (np.array): decision vectors.
        
        Returns the predicted means and the standard deviations. 
        """
        Xtest = (x-self.xbar)/self.xstd
        y, C = self.model.predict(Xtest)
        y_pred = y*self.ystd + self.ybar
        inds = [i for i in range(x.shape[0]) \
                if np.any(np.all(np.abs(x[i] - self.X) <= 1e-9, axis=1))]
        C[inds] = 0.0
        count = 0
        if np.any(C<0):
            print('Negative variance.')
            print(C)
            print('Model status:')
            print([opt_run.status for opt_run in\
                    self.model.optimization_runs])                
            print('Model parameters:')
            print(self.model.param_array)
            print('Parameters from optimization runs:')
            print([opt_run.x_opt for opt_run in\
                    self.model.optimization_runs])                                    
            C = np.ones(y.shape)*1e-4
        std_dev = np.sqrt(C)*self.ystd
        return y_pred, std_dev
      
    def expected_improvement(self, x, obj_sense=1, lb=None, ub=None,\
                                cfunc=None, cargs=(), ckwargs={}):
        """
        Calculate the expected improvement at a given set of decision vectors,
        based on the trained model. See the paper for details.
        
        Parameters.
        -----------
        x (np.array): a set of decision vectors.
        obj_sense (int): whether to maximise or minimise. Key for the input:
                     1: maximise
                    -1: minimise (default)
        lb (np.array): lower bound for the decision space.
        ub (np.array): upper cound for the decision space.
        cfunc (function): cheap constraint fucntion.
        cargs (tuple): arguments for the constraint function.
        ckwargs (dict): keyword arguments for the constraint function.
        
        Returns the expected improvement values at x.
        """
        if len(x.shape) < 2:
            x = np.array(x)
            x = x[:, np.newaxis].T
        if lb is not None:
            xp = np.clip(x.copy(), lb, ub)
            rows = x.shape[0]
            b_inds = [i for i in range(rows) if not np.all(xp[i]==x[i])]
            if len(b_inds) == rows: # outside boundary = zero exp. imp.
                return np.zeros((rows, 1))
            e_inds = [i for i in range(x.shape[0]) \
                if np.any(np.all(np.abs(x[i] - self.X) <= 1e-9, axis=1))]
            if len(e_inds) == rows: # violates xtol = zero exp. imp.
                return np.zeros((rows, 1))
        if cfunc is not None:
            c_inds = []
            if xp.shape[0] == 1:
                if not cfunc(xp[0], *cargs, **ckwargs):
                    return np.zeros((1,1))
            else:
                c_inds = [i for i in range(xp.shape[0]) if not cfunc(xp[i], *cargs, **ckwargs)]
                if len(c_inds) == rows:
                    return np.zeros((rows, 1))
        y, std_dev = self.predict(xp) 
        f_best = obj_sense * np.max(obj_sense * self.Y)
        u =  obj_sense * (y - f_best) / std_dev
        sinds = [i for i in range(std_dev.shape[0]) if std_dev[i] == 0] # to get rid of NaN issues
        u[sinds] = 1e30
        # normal cumulative distribution function
        PHI = (0.5 * ERF(u/np.sqrt(2.0)))+0.5 
        # normal density function
        phi = 1/np.sqrt(2.0*np.pi)*np.exp(-u**2/2.0) 
        ei =  std_dev * ((u * PHI) + (phi)) 
        ei[sinds] = 0
        if lb is not None:
            ei[e_inds] = 0
            ei[b_inds] = 0
        if cfunc is not None:
            ei[c_inds] = 0
        return ei
    
    # APR added
    
    def hammer_function_precompute(self, x0, L, Min, model):
        """
        Pre-computes the parameters of a penalizer centered at x0.
        """
        if x0 is None: return None, None
        if len(x0.shape)==1: x0 = x0[None,:]
        m = model.predict(x0)[0]
        pred = model.predict(x0)[1].copy()
        pred[pred<1e-16] = 1e-16
        s = pred # Surrogate predict outputs mu and standard deviation
        r_x0 = abs(m-Min)/L # APR added  - in strange cases the radius can go negative
        s_x0 = s/L
        r_x0 = r_x0.flatten()
        s_x0 = s_x0.flatten()
        return r_x0, s_x0
  
    def cone_function(self, x, x0):
        """
        Creates the function to define the exclusion zones.
        Using half the Lipschitz constant as the gradient of the penalizer.
        """
        r_x0 = self.r_x0
        s_x0 = self.s_x0
        x_norm = np.sqrt(np.square(np.atleast_2d(x)[:, None, :] - np.atleast_2d(x0)[None, :, :]).sum(axis=-1)) # Return the sum of the array elements over the last axis.
        norm_jitter = 1e-10
        return (x_norm + norm_jitter) / (r_x0 + s_x0)
    
    def hammer_function(self, x, x0):
        """
        The penalisation function.
        """
        conesT = self.cone_function(x, x0).T
        clipped_h_vals = np.ones(conesT.shape) 
        for i in range(len(conesT)):
            h_vals=conesT[i].reshape([1, -1])
            clipped_h = np.concatenate((h_vals, np.ones(h_vals.shape)), axis=0)                                         
            clipped_h_vals_new = np.linalg.norm(clipped_h, -5, axis=0)                                                    
            clipped_h_vals[i] = clipped_h_vals_new  
        hammer = clipped_h_vals.prod(0)
        return hammer
    
    
    def penalized_acquisition(self, x,  obj_sense=1, lb=None, ub=None, cfunc=None, cargs=(), ckwargs={}):
        '''
        Creates a penalized acquisition function using the 4th norm between
        the acquisition function and the cone.
        '''
        fval = self.expected_improvement(x, obj_sense=obj_sense, lb=lb, ub=ub, cfunc=cfunc, cargs=cargs, ckwargs=ckwargs).ravel()
        fval += 1e-50
        if self.pen_locations is not None:
            fval *= self.hammer_function(x, self.pen_locations)
        return fval
    
    
    
    # APR added
    
class MultiSurrogates(object):
    '''
    Multiple surrogates for multi-surrogate Bayesian optimisation.
    '''

    def __init__(self, xtr, ytr, kernels, verbose=False):
        '''
        This constructor creates multiple surrogates. 
        
        Parameters.
        -----------
        xtr (np.array): training decision vectors.  
        ytr (np.array): trainign objective vectors.
        kernels (GPy kernels): kernel functios to use with Gaussian processes.
        '''
        self.xtr = xtr
        self.ytr = ytr
        self.kernels = kernels
        self.n_models = len(self.ytr[0])
        self.verbose = verbose
        self.models = self.train_models()


    def train_models(self):
        '''
        Train multiple models.
        
        Returns a set of models.
        '''
        models = []
        for i in range(self.n_models):
            models.append(Surrogate(self.xtr, np.reshape(self.ytr[:,i], \
                                    (-1, 1)), self.kernels[i], verbose=self.verbose))
        return models

    def predict(self, x):
        '''
        Predict the mean objective function and the standard deviation for a 
        set of decision vectors. 
        
        Parameters. 
        -----------
        x (np.array): decision vector.
        
        Returns the mean predictions and the standard deviations.
        '''
        if len(x.shape) < 2:
            x = np.array(x)
            x = x[:, np.newaxis].T
        mean, var = [], []
        for model in self.models:
            m, v = model.predict(x)
            mean.append(m)
            var.append(v)
        return np.reshape(mean, (-1, self.n_models)), np.reshape(var, (-1, self.n_models))        
            
