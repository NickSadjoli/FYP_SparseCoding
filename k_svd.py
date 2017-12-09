import sys
import math
import numpy as np
import tables as tb
from y_test import y_test as y
from mp_functions import *

from utils import file_create
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV    

def l2_norm(v):
    return np.linalg.norm(v) / np.sqrt(len(v))

def print_sizes(Phi,y):
    print np.shape(Phi), np.shape(y)
    sys.exit(0)

def element_sum(v):
    m, n = np.shape(v)
    s = 0
    square = 0
    for i in range(0,m):
        for j in range(0,n):
            s+=v[i][j]
            square += (v[i][j])**2

    return s, square


max_iter = None
if len(sys.argv) == 4:
    chosen_mp = sys.argv[1]
    f_name = sys.argv[2]
    sparsity = int(sys.argv[3])
elif len(sys.argv) == 5:
    chosen_mp = sys.argv[1]
    f_name = sys.argv[2]
    sparsity = int(sys.argv[3])
    max_iter = sys.argv[4]
else:
    print "Please give arguments with the form of:  [MP_to_use]  [output_file_name(wo/ '.h5' prefix)]  [#mp_sparsity]  [optional\ #max_iterations(used in the MP)]"
    sys.exit(0)

mp_process = None

if chosen_mp == 'omp':
    mp_process = omp
elif chosen_mp == 'bomp':
    mp_process = bomp
elif chosen_mp == 'cosamp':
    mp_process = cosamp
elif chosen_mp == 'bmpnils':
    mp_process = bmpnils
elif chosen_mp == 'omp-scikit':
    mp_process = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity)
else: 
    print "Invalid MP chosen, please input a valid MP!"
    sys.exit(0)

vbose = input("Verbose? ==> ")

#take y from reading the .h5 file 
y_file = tb.open_file("y_large.h5", 'r')
y = y_file.root.data[:]
y_file.close()


#take Phi from reading the other .h5 file as well.
file = tb.open_file("Phi_large.h5", 'r')
Phi = file.root.data[:]
file.close()
n = np.shape(Phi)[1]
#print_sizes(Phi, y)

#Phi = np.random.normal(0, 0.5, [m,n])

Phi_init = np.zeros(Phi.shape)

Phi_init[:] = Phi[:]

tol = 1e-4

if np.shape(y)[0] <= 1000:

    for i in range(0,1000):
        Phi_old = np.zeros(Phi.shape)
        Phi_old[:] = Phi[:]
        #print Phi

        # find approximation of x signal
        if chosen_mp == "omp-scikit":
            #print np.shape(Phi), np.shape(y)
            #print sparsity 
            mp_process = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity, tol=tol)
            mp_process.fit(Phi, y)
            x_mp = mp_process.coef_
        elif max_iter is None:
            x_mp, _, _ = mp_process(Phi, y, ncoef=sparsity, verbose=vbose)
        else:
            x_mp, _, _ = mp_process(Phi, y, ncoef=sparsity, maxit=max_iter, verbose=vbose)


        '''
        # Update dictionary (Phi)
        '''

        #for every column in Phi...
        for k in range(0, n):
            #pick the kth column from Phi
            atom = Phi[:,k]

            #make a copy of Phi, with the atom column ZEROED
            copy = np.zeros(Phi.shape)
            copy[:] = Phi[:]
            copy[:,k] = 0   

            #compute error between y and dot product of Phi's copy and x_mp
            #print np.shape(copy), np.shape(x_mp)
            
            y_noatom = np.dot(copy,x_mp)
            y_noatom = np.reshape(y_noatom, (len(y_noatom),1))
            #print "Y: ", np.shape(y_noatom)
            #Error = y - np.dot(copy, x_mp)
            Error = y - y_noatom

            '''
            should be restricting error to only columns corresponding to atom, but for this case, Error would have only 1 column anyways.
            apply Full SVD decomp on Error vector
            '''
            #print "Error:", np.shape(Error)
            U,s,V = np.linalg.svd(Error, full_matrices=True)
            ''' 
                For a (100x1) y with a Phi size of (100x256), the sizes of: 
                U => (100 x 1), s => (1 x 1), V => (1 x 1)
            '''
            # update kth atom of Phi = 1st column of U
            Phi[:,k] = U[:,0] 


            '''
            update k-th value of x_mp by multiplying s and V 
            (Note that in the case of y with m x p and NOT a m x 1 size, it should be multiplication of the 
            1st element of s with the 1st column of V (i.e. s[0,0] * V[:,0])
            '''
            x_mp[k] = np.dot(s[0], V[:,0])

        #print Phi

        #print Phi_old 

        #print Phi-Phi_old
        #detected_norm = l2_norm(Phi - Phi_old)

        previous_norm = l2_norm(Phi_old)
        detected_norm = l2_norm(Phi)

        #print previous_norm, detected_norm  

        norm_diff = previous_norm - detected_norm

        '''
        sumation1, squares1 = element_sum(Phi_old)
        sumation_new, squares_new = element_sum(Phi)
        print sumation1, squares1
        print sumation_new, squares
        '''


        #if l2_norm(Phi - Phi_old) < tol:
        if abs(norm_diff) < tol:
            print abs(norm_diff), "converged"
            break

        print "Updated for: ", i , "-th iteration. Current norm = ", l2_norm(Phi-Phi_old) 


else: 
    m = np.shape(Phi)[0]
    length_y = np.shape(y)[0] / m
    #y = np.reshape(y, (m, length_y))
    y = np.reshape(y, (length_y, m)) #reshape y into length_y * smaller signals
    for i in range(0, 1000):

        Phi_old = np.zeros(Phi.shape)
        Phi_old = Phi[:]
        k = np.shape(y)[0]

        x_mp = np.zeros((k,n))

        #get approximation of x (n x k) via MP for each column in (m x k) signal
        for i in range(k):

            #get each column sample of y
            #column = y[:,i]
            column = y[i]
            # find approximation of x signal for each column
            if chosen_mp == "omp-scikit":
                #print np.shape(Phi), np.shape(y)
                #print sparsity 
                mp_process = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity, tol=tol)
                mp_process.fit(Phi, column)
                #x_mp[:,i] = mp_process.coef_
                x_mp[i] = mp_process.coef_
            elif max_iter is None:
                #x_mp[:,i], _, _ = mp_process(Phi, column, ncoef=sparsity, verbose=vbose)
                x_mp[i], _, _ = mp_process(Phi, column, ncoef=sparsity, verbose = vbose)
            else:
                #x_mp[:,i], _, _ = mp_process(Phi, column, ncoef=sparsity, maxit=max_iter, verbose=vbose)
                x_mp[i], _, _ = mp_process(Phi, column, ncoef=sparsity, maxit=max_iter, verbose = vbose)


        #for every column in Phi...
        print n
        for k in range(0, n):
            #pick the kth column from Phi
            atom = Phi[:,k]

            #make a copy of Phi, with the atom column ZEROED
            copy = np.zeros(Phi.shape)
            copy[:] = Phi[:]
            copy[:,k] = 0   

            #compute error between y and dot product of Phi's copy and x_mp
            #print np.shape(copy), np.shape(x_mp)
            #print_sizes(copy, x_mp)
            
            y_noatom = np.dot(x_mp, copy.T)
            print "gotten no_atom contribution"
            print np.shape(y_noatom)
            sys.exit(0)
            #y_noatom = np.reshape(y_noatom, (len(y_noatom),1))
            #print "Y: ", np.shape(y_noatom)
            #Error = y - np.dot(copy, x_mp)
            Error = y - y_noatom

            '''
            should be restricting error to only columns corresponding to atom, but for this case, Error would have only 1 column anyways.
            apply Full SVD decomp on Error vector
            '''
            #print "Error:", np.shape(Error)
            U,s,V = np.linalg.svd(Error, full_matrices=True)
            ''' 
                For a (100x1) y with a Phi size of (100x256), the sizes of: 
                U => (100 x 1), s => (1 x 1), V => (1 x 1)
            '''
            # update kth atom of Phi = 1st column of U
            Phi[:,k] = U[:,0] 


            '''
            update k-th value of x_mp by multiplying s and V 
            (Note that in the case of y with m x p and NOT a m x 1 size, it should be multiplication of the 
            1st element of s with the 1st column of V (i.e. s[0,0] * V[:,0])
            '''
            x_mp[k] = np.dot(s[0], V[:,0])

        previous_norm = l2_norm(Phi_old)
        detected_norm = l2_norm(Phi)

        #print previous_norm, detected_norm  

        norm_diff = previous_norm - detected_norm

        '''
        sumation1, squares1 = element_sum(Phi_old)
        sumation_new, squares_new = element_sum(Phi)
        print sumation1, squares1
        print sumation_new, squares
        '''


        #if l2_norm(Phi - Phi_old) < tol:
        if abs(norm_diff) < tol:
            print abs(norm_diff), "converged"
            break

        print "Updated for: ", i , "-th iteration. Current norm = ", l2_norm(Phi-Phi_old) 

'''
print Phi_init
print Phi
'''
m,n = np.shape(Phi)
file_create(f_name, Phi, m ,n)
