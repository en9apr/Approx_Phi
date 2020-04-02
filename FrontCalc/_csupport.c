#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include<stdio.h>
#include "csupport.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

static char module_docstring[]= "Support methods implemented in C.";
static char compare_solutions_docstring[] = "Compare two solutinos for dominance.\n"
                                            "Parameters.\n"
                                            "- sol_a (numpy array): Solution A.\n"
                                            "- sol_b (numpy array): Solution B.\n"
                                            "- obj_sense (numpy array): Objective sense.\n"
                                            "   keys.\n"
                                            "       -1: minimisation\n"
                                            "        1: maximisation\n"
                                            "Return.\n"
                                            "- result (int): the comparison between a and b.\n"
                                            "    keys:\n"
                                            "        0: a dominates.\n"
                                            "        1: b dominates.\n" 
                                            "        2: a and b are identical.\n"
                                            "        3: a and b are mutually non-dominated.\n";

static char nond_ind_docstring[] = "Extract the non-dominated indices.\n"
                                    "Parameters.\n"
                                    "- y (r x c numpy array): Containing r solutions in c-dimensinal objective space.\n"
                                    "- obj_sense (numpy array): Objective sense.\n"
                                    "   keys.\n"
                                    "       -1: minimisation\n"
                                    "        1: maximisation\n"
                                    "Return.\n"
                                    "- result (1 x r numpy array): if zero then non-dominated, else dominated.\n";
                                                                      
                                            
static PyObject *csupport_compare_solutions(PyObject *self, PyObject *args);
static PyObject *csupport_nond_ind(PyObject *self, PyObject *args);

// method definitions
static PyMethodDef module_methods[] = {
    {"compare_solutions", csupport_compare_solutions, METH_VARARGS, compare_solutions_docstring},
    {"nond_ind", csupport_nond_ind, METH_VARARGS, nond_ind_docstring},
    {NULL, NULL, 0, NULL}
};

// init function
PyMODINIT_FUNC PyInit__csupport(void)
{
    PyObject *module;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_csupport",
        module_docstring,
        -1,
        module_methods,
        NULL,
        NULL,
        NULL,
        NULL
    };
    module = PyModule_Create(&moduledef);
    if (!module)
        return NULL;

    /* Load `numpy` functionality. */
    import_array();
    
    return module;
}

// define function explcitly
static PyObject *csupport_nond_ind(PyObject *self, PyObject *args)
{
    //int ncols;
    PyObject *front, *obj_sense;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OO", &front, &obj_sense))
        return NULL;
    
    
    /* Interpret the input objects as numpy arrays. */
    
    PyObject *front_array = PyArray_FROM_OTF(front, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *obj_sense_array = PyArray_FROM_OTF(obj_sense, NPY_DOUBLE,
                                            NPY_IN_ARRAY);
    
    /* If that didn't work, throw an exception. */
    if (front == NULL || obj_sense_array == NULL) {
        Py_XDECREF(front_array);
        Py_XDECREF(obj_sense_array);
        return NULL;
    }

    /* How many data points are there? */
    int N = (int)PyArray_DIM(front_array, 0);
    int D = (int)PyArray_DIM(front_array, 1);
    
    /* Get pointers to the data as C-types. */
    double *front_ap    = (double*)PyArray_DATA(front_array);
    double *obj_sensep = (double*)PyArray_DATA(obj_sense_array);

    npy_intp dims[] = {N};    
    PyObject *ret = PyArray_SimpleNew(1, dims, NPY_LONG);
    PyArray_FILLWBYTE(ret, 0);
    
    int *ret_ap = (int*)PyArray_DATA(ret);
    extract_front_inds(front_ap, N, D, ret_ap, obj_sensep);
    /* Clean up. */
    Py_DECREF(front_array);
    Py_DECREF(obj_sense_array);
    //Py_DECREF(ret);    
    //Py_DECREF(ret_ap);   
    Py_DECREF(front_ap);   
    Py_DECREF(obj_sensep);   
    
    /* Build the output tuple */
    PyObject *nret = PyArray_SimpleNewFromData(1, dims, NPY_INT, &ret_ap[0]);
    //Py_DECREF(ret);    
    //Py_DECREF(ret_ap); 
    return nret;
}

static PyObject *csupport_compare_solutions(PyObject *self, PyObject *args)
{
    //int ncols;
    PyObject *sol_a, *sol_b, *obj_sense;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOO", &sol_a, &sol_b, &obj_sense))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *sol_a_array = PyArray_FROM_OTF(sol_a, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *sol_b_array = PyArray_FROM_OTF(sol_b, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *obj_sense_array = PyArray_FROM_OTF(obj_sense, NPY_DOUBLE,
                                            NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (sol_a_array == NULL || sol_b_array == NULL || obj_sense_array == NULL) {
        Py_XDECREF(sol_a_array);
        Py_XDECREF(sol_b_array);
        Py_XDECREF(obj_sense_array);
        return NULL;
    }

    /* How many data points are there? */
    int N = (int)PyArray_DIM(sol_a_array, 0);
    
    /* Get pointers to the data as C-types. */
    double *sol_ap    = (double*)PyArray_DATA(sol_a_array);
    double *sol_bp    = (double*)PyArray_DATA(sol_b_array);
    double *obj_sensep = (double*)PyArray_DATA(obj_sense_array);

    /* Call the external C function to compute the chi-squared. */
    int result = compare_solutions(sol_ap, sol_bp, obj_sensep, N);

    /* Clean up. */
    Py_DECREF(sol_a_array);
    Py_DECREF(sol_b_array);
    Py_DECREF(obj_sense_array);

    if (result < 0 || result > 3) {
        PyErr_SetString(PyExc_RuntimeError,
                    "Result does not make sense. Please check your installation.");
        return NULL;
    }

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("i", result);
    return ret;
}



