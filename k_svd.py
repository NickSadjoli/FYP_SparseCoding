import sys
import math
import numpy as np
from mp_functions import *

def l2_norm(v):
    return np.linalg.norm(v) / np.sqrt(len(v))

max_iter = None
if len(sys.argv) > 1:
    if len(sys.argv) == 2:
        chosen_mp = sys.argv[1]
    elif len(sys.argv) == 3:
        chosen_mp = sys.argv[1]
        max_iter = sys.argv[2]
else:
    print "Please choose an MP to use"
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


vbose = input("Verbose? ==> ")



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

m = len(y)

sparsity = 30.0

f = 2.5

n = int(math.pow(10,(m/(f*sparsity)))) #column of Phi

Phi = np.random.normal(0, 0.5, [m,n])

Phi_init = np.zeros(Phi.shape)

Phi_init[:] = Phi[:]

tol = 1e-4

for i in range(0,1000):
    Phi_old = np.zeros(Phi.shape)
    Phi_old[:] = Phi[:]

    # find approximation of x signal
    if chosen_mp == "omp-scikit":
        omp_process = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity)
        omp_process.fit(Phi, y_test)
        x_mp = omp_process.coef_
    else:
        if max_iter is None:
            x_mp, _, _ = mp_process(Phi, y, chosen_mp, ncoef=sparsity, verbose=vbose)

    '''
    # Update dictionary, Phi
    '''

    #for every column in Phi...
    for k in range(0, n+1):
        #pick the kth column from Phi
        atom = Phi[:,k]

        #make a copy of Phi, with the atom column ZEROED
        copy = np.zeros(Phi.shape)
        copy[:] = Phi[:]
        copy[:,k] = 0

        #compute error between y and dot product of Phi's copy and x_mp
        Error = y - np.dot(copy, x_mp)

        #should be restricting error to only columns corresponding to atom, but for this case, Error would have only 1 column anyways.
        #apply Full SVD decomp
        U,s,V = np.linalg.svd(Error, full_matrices=True)
        Phi[:,k] = U[:,0] # update kth atom = 1st column of U
        x_mp = np.dot(s[0,0], V[:,0]) # update x_mp (or, rather, k-th column of x_mp for y with m x p size) by multiplying 
                                      #1st element of s and 1st column of V

    if l2_norm(Phi - Phi_old) < tol:
        print "converged"
        break

print Phi_init
print Phi
