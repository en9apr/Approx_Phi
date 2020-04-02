#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Mono-Surrogate Approaches to Single- and Multi-Objective Bayesian Optimisation
================================================================================
:Author:
   Alma Rahat   <A.A.M.Rahat@exeter.ac.uk>
:Date:
   19 April 2017
:Copyright:
   Copyright (c)  Alma Rahat, University of Exeter, 2017
:File:
   mono_surrogate.py
"""

# imports
import numpy as np
import itertools
from evoalgos.performance import FonsecaHyperVolume as FH
import time
try: 
    from BO_base import BayesianOptBase
    from surrogate import Surrogate
except:
    from .BO_base import BayesianOptBase
    from .surrogate import Surrogate

# APR added
import scipy
# APR added

class MonoSurrogate(BayesianOptBase):
    """
    Mono-surrogate base class; inherits from Bayesian optimiser base class.
    """

    def __init__(self, func, n_dim, n_obj, lower_bounds, upper_bounds, \
                    obj_sense=[-1, -1], args = (), kwargs={}, X=None, Y=None, \
                    kern=None, ref_vector=None):
        """This constructor creates the mono-surrogate base class.
        
        Parameters.
        -----------
        func (method): the method to call for actual evaluations
        n_dim (int): number of decision space dimensions
        n_obj (int): number of obejctives
        lower_bounds (np.array): lower bounds per dimension in the decision space
        upper_boudns (np.array): upper bounds per dimension in the decision space
        obj_sense (np.array or list): whether to maximise or minimise each 
                objective (ignore this as it was not tested). 
                keys. 
                    -1: minimisation
                     1: maximisation
        args (tuple): the required arguments for the objective function
        kwargs (dictionary): dictionary of keyword argumets to pass to the 
                            objective function. 
        X (np.array): initial set of decision vectors.
        Y (np.array): initial set of objective values (associated with X).
        kern (GPy kernel): kernel to be used with Gaussian process.
        ref_vector (np.array): reference vector in the objective space.
        """
        super().__init__(func, n_dim, n_obj, lower_bounds, upper_bounds, \
                                obj_sense, args, kwargs, X, Y,\
                                ref_vector=ref_vector)
        self.kernel = kern
        
    def get_toolbox(self, xtr, skwargs, cfunc=None, \
                        cargs=(), ckwargs={}, verbose=True):
        """
        Generate a DEAP toolbox for the infill criterion optimiser.
        
        Parameters. 
        -----------
        xtr (np.array): traing decision vectors.
        skwargs (dict): options for infill criterion calculation; varies with 
                        technique.
        cfunc (function): cheap constraint function.
        cargs (tuple): argumetns for constraint function.
        ckwargs (dict): keyword arguments for constraint function.
        verbose (bool): whether to print more comments. 
        
        Returns a DEAP toolbox.     
        """
        ytr = self.scalarise(xtr, kwargs=skwargs)
        self.current_hv = self.current_hpv()
        surr = Surrogate(xtr, ytr, self.kernel.copy(), verbose=verbose)
        return self.init_deap(surr.expected_improvement, obj_sense=1, cfunc=cfunc, cargs=cargs, \
                        ckwargs=ckwargs, lb=self.lower_bounds, ub=self.upper_bounds)
    
class HypI(MonoSurrogate):
    '''
    Mono-surrogate Hypervolume Improvement (HypI) infill criterion.
    '''

    def __init__ (self, func, \
                    n_dim, n_obj, lower_bounds, upper_bounds, \
                    obj_sense=[-1, -1], args = (), kwargs={}, X=None, Y=None,\
                    kern=None, ref_vector=None):
        '''
        Simple constructor invoking parent.
        
        Parameters.
        -----------
        func (method): the method to call for actual evaluations
        n_dim (int): number of decision space dimensions
        n_obj (int): number of obejctives
        lower_bounds (np.array): lower bounds per dimension in the decision space
        upper_boudns (np.array): upper bounds per dimension in the decision space
        obj_sense (np.array or list): whether to maximise or minimise each 
                objective (ignore this as it was not tested). 
                keys. 
                    -1: minimisation
                     1: maximisation
        args (tuple): the required arguments for the objective function
        kwargs (dictionary): dictionary of keyword argumets to pass to the 
                            objective function. 
        X (np.array): initial set of decision vectors.
        Y (np.array): initial set of objective values (associated with X).
        kern (GPy kernel): kernel to be used with Gaussian process.
        ref_vector (np.array): reference vector in the objective space.
        '''
        super().__init__(func, n_dim, n_obj, lower_bounds, upper_bounds, \
                                obj_sense, args, kwargs, X, Y, kern=kern,\
                                ref_vector=ref_vector)
        
    def scalarise(self, x, kwargs={}):
        '''
        Hypervolume improvement computation for a given set of solutions.
        See paper for full description. 
        
        Parameters.
        -----------
        x (np.array): decision vectors.
        kwargs (dict): a dictionary of options. They are;
            'ref_vector' (np.array): reference vector
            'approximate_ref' (bool): whether to approximate reference vector 
                                    using minimum and maximum within the function 
                                    responses.
                                    
        Returns an array of hypervolume improvements.
        '''
        start = time.time()
        ref_vector = kwargs.get('ref_vector', None)
        approximate_ref = kwargs.get('approximate_ref', False)
        y = self.m_obj_eval(x)
        self.X = x
        n_data = x.shape[0]
        h = np.zeros(n_data)
        if approximate_ref:
            ref_vector = np.max(y, axis=0) + 0.1 * (np.max(y, axis=0) - np.min(y, axis=0))
            print("New Reference vector: ", ref_vector)
        y, comp_mat = self.get_dom_matrix(y, ref_vector)
        shells = []
        h_shells = []
        loc_comp_mat = comp_mat.copy()
        hpv = FH(ref_vector)
        del_inds = []   
        # shell ranking     
        while True:
            fr_inds = self.get_front(y, loc_comp_mat, del_inds)
            if fr_inds.shape[0] == 0:
                break
            shells.append(fr_inds)
            h_shells.append(hpv.assess_non_dom_front(y[fr_inds]))
            del_inds = np.concatenate([fr_inds, del_inds], axis=0)
            loc_comp_mat[:,fr_inds] = loc_comp_mat[fr_inds, :] = -1
        n_shells = len(shells)
        # hypI conputation
        for i in range(n_shells-1):
            for j in shells[i]:
                comp_row = comp_mat[j]
                # find dominated next shell indices
                nondominated = np.where(comp_row[shells[i+1]] == 3)[0]
                nfr = np.concatenate([[j], shells[i+1][nondominated]])
                h[j] = hpv.assess_non_dom_front(y[nfr])
        print('Total time: ', (time.time() - start)/60.0)
        return np.reshape(h, (-1, 1))        

        
class MSD(MonoSurrogate):
    '''
    Mono-surrogate Minimum Signed Distance (MSD) infill criterion.
    '''
    
    def __init__ (self, func, \
                    n_dim, n_obj, lower_bounds, upper_bounds, \
                    obj_sense=[-1, -1], args = (), kwargs={}, X=None, Y=None,\
                    kern=None, ref_vector=None):
        '''
        Simple constructor invoking parent. 
        
        Parameters.
        -----------
        func (method): the method to call for actual evaluations
        n_dim (int): number of decision space dimensions
        n_obj (int): number of obejctives
        lower_bounds (np.array): lower bounds per dimension in the decision space
        upper_boudns (np.array): upper bounds per dimension in the decision space
        obj_sense (np.array or list): whether to maximise or minimise each 
                objective (ignore this as it was not tested). 
                keys. 
                    -1: minimisation
                     1: maximisation
        args (tuple): the required arguments for the objective function
        kwargs (dictionary): dictionary of keyword argumets to pass to the 
                            objective function. 
        X (np.array): initial set of decision vectors.
        Y (np.array): initial set of objective values (associated with X).
        kern (GPy kernel): kernel to be used with Gaussian process.
        ref_vector (np.array): reference vector in the objective space.
        '''
        super().__init__(func, n_dim, n_obj, lower_bounds, upper_bounds, \
                                obj_sense, args, kwargs, X, Y, kern=kern,\
                                ref_vector=ref_vector)
    
    def scalarise(self, x, kwargs={}):
        """
        Minimum signed distance from the Pareto front. See paper for full details.
        
        Parameters.
        -----------
        x (np.array): decision vectors.
        kwargs (dict): not used in this case.
        
        Returns an array of distances. 
        """
        start = time.time()
        y = self.m_obj_eval(x)
        self.X = x
        n_data = x.shape[0]
        h = np.zeros(n_data)
        y, comp_mat = self.get_dom_matrix(y)
        front_inds = self.get_front(y, comp_mat)
        for i in range(n_data):
            if i not in front_inds:
                dist = [np.sum(y[k]-y[i]) for k in front_inds]
                h[i] = np.min(dist)
        print('Total time: ', (time.time() - start)/60.0)
        return np.reshape(h, (-1, 1))
        
        
class DomRank(MonoSurrogate):
    '''
    Mono-surrogate dominance ranking infill criterion.    
    '''
    
    def __init__ (self, func, \
                    n_dim, n_obj, lower_bounds, upper_bounds, \
                    obj_sense=[-1, -1], args = (), kwargs={}, X=None, Y=None,\
                    kern=None, ref_vector=None):
        '''
        Simple constructor for invoking parent class.
        
        Parameters.
        -----------
        func (method): the method to call for actual evaluations
        n_dim (int): number of decision space dimensions
        n_obj (int): number of obejctives
        lower_bounds (np.array): lower bounds per dimension in the decision space
        upper_boudns (np.array): upper bounds per dimension in the decision space
        obj_sense (np.array or list): whether to maximise or minimise each 
                objective (ignore this as it was not tested). 
                keys. 
                    -1: minimisation
                     1: maximisation
        args (tuple): the required arguments for the objective function
        kwargs (dictionary): dictionary of keyword argumets to pass to the 
                            objective function. 
        X (np.array): initial set of decision vectors.
        Y (np.array): initial set of objective values (associated with X).
        kern (GPy kernel): kernel to be used with Gaussian process.
        ref_vector (np.array): reference vector in the objective space.
        '''
        super().__init__(func, n_dim, n_obj, lower_bounds, upper_bounds, \
                                obj_sense, args, kwargs, X, Y, kern=kern,\
                                ref_vector=ref_vector)

    def scalarise(self, x, kwargs={}):
        """
        Dominance ranking infill criterion. See paper for full details.
        
        Parameters.
        -----------
        x (np.array): decision vectors.
        kwargs (dict): not used in this case.
        
        Returns an array of distances. 
        """
        start = time.time()
        y = self.m_obj_eval(x)
        self.X = x
        n_data = x.shape[0]
        h = np.zeros(n_data)
        y, comp_mat = self.get_dom_matrix(y)
        front_inds = self.get_front(y, comp_mat)
        for i in range(n_data):
            if i not in front_inds:
                row = comp_mat[i,:]
                count = np.where(row == 1)[0].shape[0]
                count = count / (n_data - 1)
                h[i] = count 
        h = 1 - h
        print('Total time: ', (time.time() - start)/60.0)
        return np.reshape(h, (-1, 1))
        

class ParEGO(MonoSurrogate):
    '''
    Mono-surrogate ParEGO.
    '''
   
    def __init__ (self, func, \
                    n_dim, n_obj, lower_bounds, upper_bounds, \
                    obj_sense=[-1, -1], args = (), kwargs={}, X=None, Y=None,\
                    kern=None, ref_vector=None):
        '''
        Simple constructor for invoking parent.
        
        Parameters.
        -----------
        func (method): the method to call for actual evaluations
        n_dim (int): number of decision space dimensions
        n_obj (int): number of obejctives
        lower_bounds (np.array): lower bounds per dimension in the decision space
        upper_boudns (np.array): upper bounds per dimension in the decision space
        obj_sense (np.array or list): whether to maximise or minimise each 
                objective (ignore this as it was not tested). 
                keys. 
                    -1: minimisation
                     1: maximisation
        args (tuple): the required arguments for the objective function
        kwargs (dictionary): dictionary of keyword argumets to pass to the 
                            objective function. 
        X (np.array): initial set of decision vectors.
        Y (np.array): initial set of objective values (associated with X).
        kern (GPy kernel): kernel to be used with Gaussian process.
        ref_vector (np.array): reference vector in the objective space.
        '''
        super().__init__(func, n_dim, n_obj, lower_bounds, upper_bounds, \
                        obj_sense, args, kwargs, X, Y, kern=kern,\
                        ref_vector=ref_vector)
                        
                        
    def normalise(self, y):
        """
        Normalise cost functions. Here we use estimated limits from data in 
        normalisation as suggested by Knowles (2006).
        
        Parameters. 
        -----------
        y (np.array): matrix of function values.
        
        Returns normalised function values.
        """
        min_y = np.min(y, axis=0)
        max_y = np.max(y, axis=0)
        return (y - min_y)/(max_y - min_y)
        
    def get_lambda(self, s, n_obj):
        """
        Select a lambda vector. See Knowles(2006) for full details. 
        
        Parameters. 
        -----------
        s (int): determine total number of vectors: from (s+k-1) choose (k-1)
                    vectors. 
        n_obj (int): number of objectvies.
        
        Returns a selected lambda vector.
        """
        try:
            self.l_set
        except:
            l = [np.arange(s+1, dtype=int) for i in range(n_obj)]
            self.l_set = np.array([np.array(i) \
                    for i in itertools.product(*l) if np.sum(i) == s])/s
            print("Number of scalarising vectors: ", self.l_set.shape[0])
        ind = np.random.choice(np.arange(self.l_set.shape[0], dtype=int))
        return self.l_set[ind]
        
    def scalarise(self, x, kwargs={}):
        """
        Transform cost functions with augmented chebyshev -- ParEGO infill 
        criterion. 
        See Knowles(2006) for full details.
        
        Parameters.
        -----------
        x (np.array): decision vectors.
        kwargs (dict): dictionary of options. They are.
                    's' (int): number of lambda vectors. 
                    'rho' (float): rho from ParEGO
                    
        Returns an array of transformed cost.
        """
        s = kwargs.get('s', 5)
        rho = kwargs.get('rho', 0.05)
        y = self.m_obj_eval(x)
        self.X = x
        y_norm = self.normalise(y)
        lambda_i = self.get_lambda(s, y.shape[1])
        new_y = np.max(y_norm * lambda_i, axis=1) + (rho * np.dot(y, lambda_i))
        return np.reshape(-new_y, (-1, 1))
        
class EGO(MonoSurrogate):
    '''
    Mono-surrogate single obejctive optimiser that uses expected improvement as
    infill criterion.
    '''
   
    def __init__ (self, func, \
                    n_dim, n_obj, lower_bounds, upper_bounds, \
                    obj_sense=-1, args = (), kwargs={}, X=None, Y=None,\
                    kern=None, ref_vector=None):
        '''
        Simple constructor for invoking parent class.
        
        Parameters.
        -----------
        func (method): the method to call for actual evaluations
        n_dim (int): number of decision space dimensions
        n_obj (int): number of obejctives
        lower_bounds (np.array): lower bounds per dimension in the decision space
        upper_boudns (np.array): upper bounds per dimension in the decision space
        obj_sense (np.array or list): whether to maximise or minimise each 
                objective (ignore this as it was not tested). 
                keys. 
                    -1: minimisation
                     1: maximisation
        args (tuple): the required arguments for the objective function
        kwargs (dictionary): dictionary of keyword argumets to pass to the 
                            objective function. 
        X (np.array): initial set of decision vectors.
        Y (np.array): initial set of objective values (associated with X).
        kern (GPy kernel): kernel to be used with Gaussian process.
        ref_vector (np.array): reference vector in the objective space.
        '''
        super().__init__(func, n_dim, n_obj, lower_bounds, upper_bounds, \
                        obj_sense, args, kwargs, X, Y, kern=kern,\
                        ref_vector=ref_vector)
    
    def get_toolbox(self, xtr, skwargs, cfunc=None, \
                        cargs=(), ckwargs={}, verbose=True):
        """
        Generate a DEAP toolbox for the infill criterion optimiser.
        
        Parameters. 
        -----------
        xtr (np.array): traing decision vectors.
        skwargs (dict): options for infill criterion calculation; varies with 
                        technique.
        cfunc (function): cheap constraint function.
        cargs (tuple): argumetns for constraint function.
        ckwargs (dict): keyword arguments for constraint function.
        verbose (bool): whether to print more comments. 
        
        Returns a DEAP toolbox.     
        """
        ytr = self.scalarise(xtr, kwargs=skwargs)
        self.current_hv = self.current_hpv()
        surr = Surrogate(xtr, ytr, self.kernel.copy(), verbose=verbose)
        
        #########################################
        # Add surrogate to base class (APR added)
        #########################################
        self.surr=surr
        
        ########################
        # Get bounds (APR added)
        ########################
        lb=self.lower_bounds
        ub=self.upper_bounds
        xmean = self.surr.xbar
        xsd = self.surr.xstd
        ysd = self.surr.ystd
        bounds = self.get_bounds((lb-xmean)/xsd, (ub-xmean)/xsd)
        
        ##################################
        # Compute the gradient (APR added)
        ##################################
        L = (self.max_predictive_gradient(surr, bounds)) * (ysd / xsd)
        self.surr.L = L

        #######################################################
        # Set best based on minimum observed so far (APR added)
        #######################################################
        self.surr.best = min(ytr)        
        
        return self.init_deap(surr.penalized_acquisition, obj_sense=self.obj_sense[0], cfunc=cfunc, cargs=cargs, \
                        ckwargs=ckwargs, lb=self.lower_bounds, ub=self.upper_bounds)


    ##############################
    # APR added Lipschitz function
    ##############################
    def max_predictive_gradient(self, surr_model, bounds):
        """
        Estimate the Lipschitz constant of f by taking maximizing the norm of the expectation of the gradient of *f*.
        """    
        def df(x,model,x0):
            x = np.atleast_2d(x)
            dmdx,_ = surr_model.model.predictive_gradients(x)
            res = np.sqrt((dmdx*dmdx).sum(1)) # simply take the norm of the expectation of the gradient
            return -res
        
        samples = self.samples_multidimensional_uniform(bounds, 500)
        samples = np.vstack([samples,surr_model.Xtr])
        pred_samples = df(samples,surr_model,0)
        x0 = samples[np.argmin(pred_samples)]
        res = scipy.optimize.minimize(df,x0, method='L-BFGS-B',bounds=bounds, args = (surr_model, x0), options = {'maxiter': 200})
        minusL = res.fun[0][0]
        L = -minusL
        if L<1e-7: L=10  ## to avoid problems in cases in which the model is flat.
        return L
    
    #############################
    # APR added sampling function
    #############################
    def samples_multidimensional_uniform(self, bounds, num_data):
        '''
        Generates a multidimensional grid uniformly distributed.
        :param bounds: tuple defining the box constraints.
        :num_data: number of data points to generate.
        '''
        dim = len(bounds)
        Z_rand = np.zeros(shape=(num_data,dim))
        #print("low",bounds[0][0],bounds[1][0])
        #print("high", bounds[0][1],bounds[1][1])
        for k in range(0,dim): Z_rand[:,k] = np.random.uniform(low=bounds[k][0],high=bounds[k][1],size=num_data)
        return Z_rand

    ###########################
    # APR added bounds function
    ###########################
    def get_bounds(self, lb, ub):
        """
        Returns a list of tuples representing bounds of the variable
        """
        return np.stack((lb,ub), axis=-1) 

                
#    def get_toolbox(self, xtr, skwargs, cfunc=None, \
#                        cargs=(), ckwargs={}, verbose=True):
#        '''
#        Modify parent's toolbox method for single objective optimisation. 
#        
#        Parameters. 
#        ----------
#        xtr (np.array): training decision vectors.
#        skwargs (dict): keyword arguments for infill criterion; not used here.
#        cfunc (function): cheap constraint function.
#        cargs (tuple): arguments of cheap constraint function. 
#        verbose (bool): whether to print verbose comments. 
#        
#        Returns a DEAP toolbox.        
#        '''
#        ytr = self.scalarise(xtr, kwargs=skwargs)
#        self.current_hv = self.current_hpv()
#        surr = Surrogate(xtr, ytr, self.kernel.copy(), verbose=verbose)
#        self.surr = surr
#        return self.init_deap(surr.expected_improvement, obj_sense=self.obj_sense[0], \
#                        cfunc=cfunc, cargs=cargs, ckwargs=ckwargs, \
#                        lb=self.lower_bounds, ub=self.upper_bounds)
        
    def scalarise(self, x, kwargs={}):
        """
        Single objective dummy scalarisation: just sends back the original cost 
        function values. This is here to make the framework coherent. 
        """
        y = self.m_obj_eval(x)
        self.X = x
        return y
        
        
