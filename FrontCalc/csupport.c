#include "csupport.h"
#include <stdio.h>

//#define DEBUG // uncomment for debugging

/*
A method for comapring two solutions for dominance. 

Parameters. 
- sol_a (double array)
- sol_b (double array)
- obj_sense (int array): an array representing the objective senses. 
    key:
        * -1: minimisation 
        *  1: maximisation 
- col (int): number of columns, i.e. the length of the solution.

Return.
- result (int): the comparison between a and b.
    key:
        0: a dominates.
        1: b dominates. 
        2: a and b are identical.
        3: a and b are mutually non-dominated.
*/
int compare_solutions(double* sol_a, double* sol_b, double* obj_sense, int ncols){
    int a_dom = 0; // a dominance count
    int b_dom = 0; // b dominance count
    int equal = 0; // equality count
    int i = 0;
    
    #ifdef DEBUG
        printf("Number of columns: %d\n", ncols);
        for (i=0; i<ncols; i++){
            printf("a[%d]: %f, %f\n", i, sol_a[i], sol_a[i]*obj_sense[i]);
            printf("b[%d]: %f, %f\n", i, sol_b[i], sol_b[i]*obj_sense[i]);
        }            
        printf("=================\n");
    #endif

    
    for (i = 0; i < ncols; i++){
        if (sol_a[i] * obj_sense[i] == sol_b[i] * obj_sense[i]){
            equal++;
        #ifdef DEBUG
            printf("a[%d]: %f, %f\n", i, sol_a[i], sol_a[i]*obj_sense[i]);
            printf("b[%d]: %f, %f\n", i, sol_b[i], sol_b[i]*obj_sense[i]);
            printf("equal: %d\n", equal);
        #endif
        }
        else if (sol_a[i] * obj_sense[i] > sol_b[i] * obj_sense[i]){
            a_dom++;
        #ifdef DEBUG
            printf("a[%d]: %f, %f\n", i, sol_a[i], sol_a[i]*obj_sense[i]);
            printf("b[%d]: %f, %f\n", i, sol_b[i], sol_b[i]*obj_sense[i]);
            printf("a_dom: %d\n", a_dom);
        #endif
        }
        else{
            b_dom++;
        #ifdef DEBUG
            printf("a[%d]: %f, %f\n", i, sol_a[i], sol_a[i]*obj_sense[i]);
            printf("b[%d]: %f, %f\n", i, sol_b[i], sol_b[i]*obj_sense[i]);
            printf("b_dom: %d\n", b_dom);
        #endif
        }
    }
    #ifdef DEBUG
        printf("=================\n");  
        printf("a_dom: %d\n", a_dom);
        printf("b_dom: %d\n", b_dom);
        printf("equal: %d\n", equal);
        //printf("Result: %d\n", result);
    #endif 
    int result = 0;
    if (b_dom == ncols){
        result = 1; // b dominates
    }
    else if (a_dom == ncols){
        result = 0; // a dominates
    }
    else if (equal == ncols){
        result = 2; // a and b are identical
    }
    else if (equal > 0){
        if(b_dom + equal == ncols){
            result = 1; // b doiminates
        }
        else if(a_dom + equal == ncols){
            result = 0; // a dominates
        }
        else{ 
            result = 3;
        }
    }
    else{
        result = 3; // a and b mutually non-dominated
    }
        
    #ifdef DEBUG
        printf("=================\n");  
        printf("a_dom: %d\n", a_dom);
        printf("b_dom: %d\n", b_dom);
        printf("equal: %d\n", equal);
        printf("Result: %d\n", result);
    #endif 
    
    return result;
}


void extract_front_inds(double* front, int rows, int cols, int* redundant, double* obj_sense)
{
    int i, j;
    int result = 0;
    for(i =0; i<rows; i++)
    {
        if (redundant[i] == 0)
        {
            for (j=0; j<rows; j++)
            {
                if ((i!=j) && (redundant[j]== 0))
                {
                    result = compare_solutions(&front[i*cols], &front[j*cols], obj_sense, cols);
                    if (result == 0)
                        redundant[j] = 1;
                }
            }
        }
        
    }  

}
