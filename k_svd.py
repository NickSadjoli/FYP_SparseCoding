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
    sparsity = sys.argv[3]
elif len(sys.argv) == 5:
    chosen_mp = sys.argv[1]
    f_name = sys.argv[2]
    sparsity = sys.argv[3]
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

'''
rms = [0]
R_error = [0]
rms_w_noise = [0]
R_error_w_noise = [0]
runtime = [0]
runtime_w_noise = [0]
snr_values = [0]
'''

y_file = tb.open_file("y_mini.h5", 'r')
y = y_file.root.data[:]
y_file.close()

vbose = input("Verbose? ==> ")

'''
y = [ 7.14872976, 7.99683485, 6.77462035, 7.35682238, 3.46190329, 8.55134831, 1.23216904, 8.02565473, \
2.48947843, 0.4399024, 3.55781487, 2.50578499, 1.03091104, 7.78104952, 9.40652756, 3.77218784, \
8.71719537, 2.04096106, 3.91879149, 1.12852764, 4.11943121, 0.8026437, 4.27436005, 2.22510807, \
4.24192647, 5.13894762, 7.97346514, 4.25494931,0.18219066,1.03641905, 4.92143153, 6.45918122, \
4.45901416, 4.88407393, 2.09664697, 9.26577278, 3.23315769, 5.10595863, 0.39743349, 6.85745321, \
5.80064463, 4.08567124, 8.73795576, 1.6692493, 4.15961993, 8.07245449, 0.7881806, 0.49578293, \
1.67697071, 0.35945209, 4.66053785, 3.60989746, 7.54721211, 0.43692716, 8.04713456, 4.32884169, \
9.95651014, 0.40068128, 4.97639291, 2.48643936, 5.73888652, 8.22627389, 9.95704624, 0.67972128, \
5.80171823, 8.4017826 , 8.39536605, 5.58673626, 7.46459042, 8.66673671, 0.38849337, 8.84118978, \
7.12208901, 7.28906882, 7.0982191 , 7.45902509, 2.12682096, 1.72751311, 5.48903346, 2.61131723, \
3.02831813, 3.32375689, 4.18218502, 4.86000617, 9.72692941, 3.16326772, 0.75072162, 2.27796348, \
2.71191956, 7.72377838, 2.77066899, 1.76803678, 2.80066975, 6.1347283 , 8.64868878, 5.53528607, \
0.5610447 , 2.65741932, 4.14732198, 4.53096657, 5.60758129, 2.58230706, 2.56200613, 5.76862782, \
1.86675798, 4.40260582, 0.45601149, 0.94533482, 1.76847433, 1.50444219, 7.52284895, 4.43420321, \
2.02678887, 5.48042672, 8.10572836, 6.7875623 , 4.84552806, 2.76148075, 9.47333587, 4.93576891, \
1.23842482, 0.8383871 , 1.63126236, 3.65153318, 2.78567095, 9.52601736, 4.68414818,5.85502263]
'''

#print np.shape(y)

#y = np.reshape(y, [50, 200]) #reshape y so that it can be manipulated using SVD

'''
m = len(y)
f = 2.5
n = int(math.pow(10,(m/(f*float(sparsity))))) #column of Phi
'''

#sparsity = 30


file = tb.open_file("Phi_mini.h5", 'r')
Phi = file.root.data[:]
file.close()
n = np.shape(Phi)[1]

#Phi = np.random.normal(0, 0.5, [m,n])

Phi_init = np.zeros(Phi.shape)

Phi_init[:] = Phi[:]

tol = 1e-4

for i in range(0,1000):
    Phi_old = np.zeros(Phi.shape)
    Phi_old[:] = Phi[:]
    #print Phi

    # find approximation of x signal
    if chosen_mp == "omp-scikit":
        print np.shape(Phi), np.shape(y)
        print sparsity
        mp_process = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity, tol=tol)
        mp_process.fit(Phi, y)
        x_mp = mp_process.coef_
    elif max_iter is None:
        x_mp, _, _ = mp_process(Phi, y, ncoef=sparsity, verbose=vbose)
    else:
        x_mp, _, _ = mp_process(Phi, y, ncoef=sparsity, maxit=max_iter, verbose=vbose, )

    '''
    # Update dictionary, Phi
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

    print("Updated for: ", i , "-th iteration. Current norm = ", detected_norm)

'''
print Phi_init
print Phi
'''
m,n = np.shape(Phi)
file_create(f_name, Phi, m ,n)
