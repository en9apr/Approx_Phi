#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Multi-Surrogate Approaches to Multi-Objective Bayesian Optimisation
================================================================================
:Author:
   Alma Rahat   <A.A.M.Rahat@exeter.ac.uk>
:Date:
   19 April 2017
:Copyright:
   Copyright (c)  Alma Rahat, University of Exeter, 2017
:File:
   multi_surrogate.py
"""
# imports
import numpy as np
from scipy.special import erf as ERF
from scipy.stats import norm as NORM
try:
    from surrogate import MultiSurrogates
except:
    from .surrogate import MultiSurrogates
import _csupport as CS
from evoalgos.performance import FonsecaHyperVolume as FH
try:
    from BO_base import BayesianOptBase
except:
    from .BO_base import BayesianOptBase


class MultiSurrogate(BayesianOptBase):
    '''
    Base class for multi-surrogate approaches.
    '''

    def __init__(self, func, \
                    n_dim, n_obj, lower_bounds, upper_bounds, \
                    obj_sense=[-1, -1], args = (), kwargs={}, X=None, Y=None, \
                    kern=None, ref_vector=None):
        '''
        This constructor creates the multi-surrogate base class.
        
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
                                obj_sense, args, kwargs, X, Y,\
                                ref_vector=ref_vector)
        self.kernels = kern
        self.hpv = FH(self.ref_vector)
        
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
        self.budget = skwargs.get('budget', 250)
        self.update(xtr) # update various parameters of multi-surrogates
        return self.init_deap(self.scalarise, cfunc=cfunc, cargs=cargs, \
                        ckwargs=ckwargs)
        
    def build_models(self, xtr, ytr, kernels):
        '''
        Build multiple surrogate models.
        
        Parameters. 
        -----------
        xtr (np.array): training decision vectors.
        ytr (np.array): training objective vectors.
        kernels (list): GPy kernels to be used with multiple Gaussian processes. 
        
        Returns multiple regression models.
        '''
        self.xtr = xtr
        self.ytr = ytr
        models = MultiSurrogates(xtr, ytr, kernels)
        return models
    
    def update(self, x_new):
        '''
        Update a range of setting for multi-surrogate.
        
        Parameters. 
        -----------
        x_new (np.array): decision vector.
        
        Updates attributes, but returns nothing. 
        '''
        # set/update xtr and ytr
        self.xtr = x_new
        self.ytr = self.m_obj_eval(x_new)
        assert self.xtr.shape[0] == self.ytr.shape[0] # check for smae amount of
                                                      # data
        yt, comp_mat = self.get_dom_matrix(self.ytr)
        self.comp_mat = comp_mat
        # update budget count
        self.b_count = self.budget - len(self.xtr) - 1 
        # update + optimise models
        self.models = self.build_models(self.xtr, self.ytr, \
                            [kern.copy() for kern in self.kernels])
        # current pf
        self.pfr_inds = self.get_front(self.ytr, self.comp_mat)
        # current hv
        self.current_hv = self.current_hpv()
        # epsilon for sms-ego
        n_pfr = len(self.pfr_inds)
        c = 1 - (1/ 2**self.n_obj)
        self.epsilon = (np.max(self.ytr, axis=0) - np.min(self.ytr, axis=0))\
                        /(n_pfr + (c * self.b_count))
                        
    
        
class MPoI(MultiSurrogate):
    '''
    Multi-Surrogate Minimum Probability of Improvement infill Criterion.
    '''

    def __init__(self, func, \
                    n_dim, n_obj, lower_bounds, upper_bounds, \
                    obj_sense=[-1, -1], args = (), kwargs={}, X=None, Y=None, \
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
        kern (GPy kernel): kernels to be used with Gaussian process.
        ref_vector (np.array): reference vector in the objective space.
        '''
        super().__init__(func, n_dim, n_obj, lower_bounds, upper_bounds, \
                                obj_sense, args, kwargs, X, Y,\
                                kern=kern, ref_vector=ref_vector)
        
    def scalarise(self, x, cfunc=None, cargs=(), ckwargs={}):
        '''
        Calculate the minimum probability of improvement compared to current 
        Pareto front. Refer to the paper for full details.
        
        parameters:
        -----------
        x (np.array): decision vectors.
        cfunc (function): cheap constraint function.
        cargs (tuple): argument for constraint function.
        ckwargs (dict): keyword arguments for constraint function.
        
        Returns scalarised cost.
        '''
        n_sols = x.shape[0]
        if cfunc is not None:
            if not cfunc(x, *cargs, **ckwargs):
                return np.zeros((1,1))  # penalise for constraint violation
        yp, stdp = self.models.predict(x)
        y = self.ytr[self.pfr_inds]
        res = np.zeros((yp.shape[0], 1))
        sqrt2 = np.sqrt(2)
        for i in range(yp.shape[0]):
            m = (yp[i] - y)/(sqrt2 * stdp[i])
            pdom = 1 - np.prod(0.5 * (1 + ERF(m)), axis=1)
            res[i] = np.min(pdom)
        return res
        
class SMSEGO(MultiSurrogate):
    '''
    Multi-surrogate SMS-EGO. Note that this is inspired from GPareto package in 
    R.
    '''
    
    def __init__(self, func, \
                    n_dim, n_obj, lower_bounds, upper_bounds, \
                    obj_sense=[-1, -1], args = (), kwargs={}, X=None, Y=None, \
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
        kern (GPy kernel): kernels to be used with Gaussian process.
        ref_vector (np.array): reference vector in the objective space.
        '''
        super().__init__(func, n_dim, n_obj, lower_bounds, upper_bounds, \
                                obj_sense, args, kwargs, X, Y,\
                                kern=kern, ref_vector=ref_vector)
        self.gain = - NORM.ppf(0.5 * (0.5**(1/self.n_obj)))

    def compare_add_solution(self, y, ytest, obj_sense):
        '''
        Compare and add a solution to the data set given its not dominated.
        
        Parameters. 
        -----------
        y (np.array): current Pareto front objective vectors.
        y_test (np.array): candidate for adding to the archive.
        
        Returns latest Pareto front.
        '''
        result = np.ones(y.shape[0])
        for i in range(y.shape[0]):
            result[i] = CS.compare_solutions(y[i], ytest, self.obj_sense)
            if result[i] == 0:
                return y
        inds = np.where(result==3)[0]
        try:
            return np.concatenate([y[inds], ytest])
        except ValueError:
            print("Likely error in y: ", y[inds])
            return ytest
                        
    def penalty(self, y, y_test):
        '''
        Penalty mechanism in the infill criterion. Penalise if dominated by the 
        current front.
        
        Parameters. 
        -----------
        y (np.array): current Pareto front elements.
        y_test (np.array): tentative solution.
        
        Returns a penalty value.
        '''
        yt = y_test - (self.epsilon * self.obj_sense)
        l = [-1 + np.prod(1 + y_test - y[i]) \
                if CS.compare_solutions(y[i], yt, self.obj_sense) == 0\
                else 0 for i in range(y.shape[0])]
        return (max([0, max(l)]))
        
    def scalarise(self, x, cfunc=None, cargs=(), ckwargs={}):
        '''
        S-metric infill criterion.
        
        parameters:
        -----------
        x (np.array): decision vectors.
        cfunc (function): cheap constraint function.
        cargs (tuple): argument for constraint function.
        ckwargs (dict): keyword arguments for constraint function.
        
        Returns scalarised cost.
        '''
        n_sols = len(x)
        if cfunc is not None:
            if not cfunc(x, *cargs, **ckwargs):
                return np.ones((1,1))*-100 # heavily penalise for constraint
                                           # violation
        # predictions
        yp, stdp = self.models.predict(x)
        # lower confidence bounds
        yl = yp - (self.gain * np.multiply(self.obj_sense, stdp))
        pen = self.penalty(self.ytr[self.pfr_inds], yl)
        if pen > 0:
            return np.array([-pen])            
        # new front
        yn = self.compare_add_solution(self.ytr[self.pfr_inds], yl, self.obj_sense)
        return np.array([self.hpv.assess_non_dom_front(yn) - self.current_hv])


