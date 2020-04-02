# supporting methods

# imports
import numpy as np
import xml.etree.ElementTree as etree

# CSV-Numpy operations
def save_to_csv(filename, numpy_array):
    np.savetxt(filename, numpy_array, delimiter=',')
    print('Array saved in: ', filename)
    
def load_from_csv(filename):
    numpy_array = np.genfromtxt(filename, delimiter=',')
    print('Initial design loaded from: ', filename)
    return numpy_array
    
def load_from_text(filename):
    f = open(filename, 'r')
    text = f.readlines()
    f.close()
    a = []
    for i in text:
        a.append(np.array(i.split(), dtype="float"))
    return np.array(a)
    
def extract_model_tags(model):
    tags = [name for name in model.parameter_names_flat()]
    tags.append(model.parameter_names()[-1])
    return tags
    
def extract_model_params(model):
    return model.param_array.tolist()
    
def counted(fn):
    def wrapper(*args, **kwargs):
        wrapper.called+= 1
        return fn(*args, **kwargs)
    wrapper.called= 0
    wrapper.__name__= fn.__name__
    return wrapper
    
def read_settings(xml_file_name):
    tree = etree.parse(xml_file_name)
    

    
    
