from IPython.display import clear_output
# example set up
import numpy as np
# import optimiser codes
import IscaOpt

settings = {\
    'n_dim': 1,\
    'n_obj': 1,\
    'lb': 0.3*np.ones(1),\
    'ub': 1.4*np.ones(1),\
    'method_name': 'EGO',\
    'budget':14,\
    'n_samples':6,\
    'visualise':True,\
    'multisurrogate':False,\
    'init_file':'initial_samples.npz'} # APR changed: n_dim, n_obj, lb, ub, deleted ref_vector, method to EGO from HypI, budget, n_samples

# APR removed benchmark

# APR added
def Forrester_Function(x):
    return np.sin(12*x - 4)*(6*x - 2)**2
# APR added



# optimise
res = IscaOpt.Optimiser.EMO(Forrester_Function, settings=settings) # APR changed function, removed args
clear_output()