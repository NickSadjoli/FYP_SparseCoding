import sys
import math
import numpy as np
import tables as tb

from mp_functions import *
import threading

from utils import file_create
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV    

def l2_norm(v):
    return np.linalg.norm(v) / np.sqrt(len(v))

def print_sizes(Phi,y):
    print np.shape(Phi), np.shape(y)
    sys.exit(0)


#process to learn and update Phi for (assumed) vectoral y (y is m x 1 sized)
def Phi_learner(Phi_list, y_list, index=None):
    global mp_process
    global m
    global n

    #create empty skeleton variable with same size as the Phi slices
    Phi = np.zeros((m,n))

    #If no index input, then indicates the inputted Phi and y are actual objects already, not a list!
    if index is None: 
        Phi = Phi_list[:]
        y = y_list

    #else, then that means inputted Phi and y are lists, and thus need to pick individual bject first.
    else:
        Phi[:] = Phi_list[index][:]
        y = np.zeros(np.shape(y_list[:, index]))
        y[:] = y_list[:, index]

        y = np.reshape(y, (len(y), 1)) #making sure to stay consistent for mp calculations 
    
    '''
    print the received sizes of y and Phi for current thread for easier debugging.
    Use of print lock required to avoid print output that clashes with outputs from other 
    thread(s)'s print's running in parallel as well 
    '''
    with print_lock:
        print threading.current_thread().name, ":", np.shape(y), np.shape(Phi)

    #prepare previous_norm variable to make iterations in later loop more effective
    previous_norm = None
    
    for i in range(0,1000):
        Phi_old = np.zeros(Phi.shape)
        Phi_old[:, :] = Phi[:, :]
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


        
        '''Update dictionary (Phi) '''

        #for every atom (k-th atom) in Phi...
        for k in range(0, n):

            #make a copy of Phi, with the k-th atom column ZEROED
            copy = np.zeros(Phi.shape)
            copy[:] = Phi[:]
            copy[:,k] = 0   

            #compute error between y and dot product of Phi's copy and x_mp

            y_noatom = np.dot(copy,x_mp)
            y_noatom = np.reshape(y_noatom, (len(y_noatom),1))

            Error = y - y_noatom

            '''
            # Restrict error to be contribution from chosen atom. 
            # In this case, error would have size of one column only.
            # Hence apply full SVD decomp on Error vector.

            # Example of SVD result: 
                For a (100x1) y with a Phi size of (100x256), the sizes of: 
                U => (100 x 1), s => (1 x 1), V => (1 x 1)
            '''

            U,s,V = np.linalg.svd(Error, full_matrices=True)
            
            # update kth atom of Phi = 1st column of U
            Phi[:,k] = U[:,0] 

            #update the k-th row of x_mp to be product of first column from V with first element value of s
            x_mp[k] = np.dot(s[0], V[:,0])


        #calculate l2_norm from previous and current Phi, and their differences
        if previous_norm is None:
            previous_norm = l2_norm(Phi_old)

        current_norm = l2_norm(Phi)

        norm_diff = previous_norm - current_norm

        #update previous_norm for next iteration's convergence testing
        previous_norm = current_norm

        #if norm difference is less than tol value, then Phi has already converged
        if abs(norm_diff) < tol:
            with print_lock:
                print threading.current_thread().name, ":" , abs(norm_diff), "converged"

            #return value of Phi back to caller.
            return Phi

        #print current thread iteration for visual debug. Note use of print lock as explained previously
        with print_lock:
            print threading.current_thread().name, ": Updated for ->", i, "-th iteration. Current norm =", l2_norm(Phi-Phi_old)


#process to learn and update Phi for y in matrix value
def Phi_learner_multi_x(Phi, y_mp):

    global mp_process
    global n
    for i in range(0, 10):

        Phi_old = np.zeros(Phi.shape)
        Phi_old[:, :] = Phi[:, :]
        #print Phi_old
        #k = np.shape(y)[0]
        g = np.shape(y_mp)[1]
        #k = np.shape(y)[1]

        x_mp = np.zeros((n,g)) #shape x to have a shape of n rows of k mp_result signals
        #x_mp = np.zeros((k,n))
        #print np.shape(x_mp)
        #sys.exit(0)

        #get approximation of x via MP for each column in (m x k) signal
        #print np.shape(y_mp)
        for j in range(g):
        #for i in range(len(y), length_y):
            #get each column sample of y
            column = y_mp[:,j]
            #column = y[i]
            # find approximation of x signal for each column
            if chosen_mp == "omp-scikit":
                #print np.shape(Phi), np.shape(y)
                #print sparsity 
                mp_process = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity, tol=tol)
                mp_process.fit(Phi, column)
                x_mp[:,j] = mp_process.coef_
                #x_mp[i] = mp_process.coef_
            elif max_iter is None:
                x_mp[:,j], _, _ = mp_process(Phi, column, ncoef=sparsity, verbose=vbose)
                #x_mp[i], _, _ = mp_process(Phi, column, ncoef=sparsity, verbose = vbose)
            else:
                x_mp[:,j], _, _ = mp_process(Phi, column, ncoef=sparsity, maxit=max_iter, verbose=vbose)
                #x_mp[i], _, _ = mp_process(Phi, column, ncoef=sparsity, maxit=max_iter, verbose = vbose)

        #print np.shape(x_mp)
        #sys.exit(0)

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
            #y_noatom = np.dot(x_mp, copy.T)
            y_noatom = np.dot(copy, x_mp)

            #Error = y - np.dot(copy, x_mp)
            Error = y_mp - y_noatom

            '''
            should be restricting error to only columns corresponding to atom, but for this case, Error would have only 1 column anyways.
            apply Full SVD decomp on Error vector
            '''
            #print "Error:", np.shape(Error)
            U,s,V = np.linalg.svd(Error, full_matrices=True)

            # update kth atom of Phi = 1st column of U
            Phi[:,k] = U[:,0] 

            '''
            update k-th value of x_mp by multiplying s and V 
            (Note that in the case of y with m x p and NOT a m x 1 size, it should be multiplication of the 
            1st element of s with the 1st column of V (i.e. s[0,0] * V[:,0])
            '''
            #print_sizes(x_mp, V)
            x_mp[k] = np.dot(s[0], V[:,0])

        previous_norm = l2_norm(Phi_old)
        detected_norm = l2_norm(Phi)

        #print previous_norm, detected_norm  

        norm_diff = previous_norm - detected_norm

        
        if abs(norm_diff) < tol:
            print abs(norm_diff), "converged"
            break
        
        '''
        difference = l2_norm(Phi-Phi_old)
        if difference < tol:
            print difference, "converged"
            break
        '''
        print "Updated for: ", i, "-th iteration. Current norm = ", detected_norm
        #print "Updated for: ", i , "-th iteration. Current norm = ", difference



print_lock = threading.Lock()

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
file = tb.open_file("Phi_small.h5", 'r')
Phi = file.root.data[:]
file.close()
m, n = np.shape(Phi)
#print_sizes(Phi, y)

#Phi = np.random.normal(0, 0.5, [m,n])

Phi_init = np.zeros(Phi.shape)

Phi_init[:] = Phi[:]

tol = 1e-4

if np.shape(y)[0] <= 1000:
    Phi_learner(Phi, y)
     

else: 
    #divide y into smaller len(y)/m pieces of sizes m
    length_y = np.shape(y)[0] / m 

    #reshape y into length_y rows of m small signals
    y = np.reshape(y, (length_y, m)) 

    #transpose y first so that the m sinals are put on the side so for more accurate recovery by Phi (y=[mxk])
    y_mp = y.T 
    g = np.shape(y_mp)[1]
    Phi_all = [Phi] * length_y

    ''' 
    #printing for debug purposes

    print np.shape(Phi_all)
    threads = [None] * length_y
    print np.shape(y_mp[:,0])
    '''

    ''' ##method that works so far, but slow (try to get it work better with threading?)
    For each Phi in Phi_all, learn a better Phi via Phi_learner (basically putting it through the actual K-SVD process)
    '''
    for index in range(0,g):
        Phi_all[index] = Phi_learner(Phi_all, y_mp, index)


    Phi = Phi_all




#for super large y signals

print m, n, len(Phi)
'''
for x in Phi:
    print x
    print '######'
'''

file_create(f_name, Phi, m ,n)
