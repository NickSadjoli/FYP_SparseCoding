from __future__ import division
import sys 
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import signaltonoise
from scipy.signal import argrelextrema
from pylab import * 
from PyAstronomy import pyasl

from sklearn.preprocessing import normalize
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.datasets import make_sparse_coded_signal
from sklearn.decomposition import DictionaryLearning
import numpy as np


from mp_functions import * #mp_functions contains all the MP algorithms to be tested.
#from Phi import Phi
import matplotlib.pyplot as plt
import unittest
import math
import tables as tb

import operator


def SNR_Test(signal):
    max_index, max_value = max(enumerate(signal), key=operator.itemgetter(1))
    leftsignal = signal[0:max_index];
    rightsignal = signal[max_index:];

    leftMin = array(leftsignal);
    rightMin = array(rightsignal);

    findLMin = argrelextrema(leftMin, np.less)[0][-1];
    findRMin = argrelextrema(rightMin, np.less)[0][0]+len(leftsignal);

    x = np.linspace(0, 100,len(signal));


    Anoise = abs(np.mean(list(signal[0:findLMin])+list(signal[findRMin:])))
    #Asignal = 1-(signal[findLMin]+signal[findRMin])/2
    Asignal = 1-Anoise;

    print (Asignal,Anoise)

    snr_value = 20*np.log10(Asignal/Anoise);

    '''    
    plot(x[0:findLMin], signal[0:findLMin],'b')
    plot(x[findLMin:findRMin],signal[findLMin:findRMin],'r')
    plot(x[findRMin:],signal[findRMin:],'b');
    plot([x[max_index], x[max_index]],[1, 1- Asignal],'r--');
    plot(x, x*0+Anoise,'b--');
    '''
    
    #plot(list(xrange(len(signal))), signal,'b')
    show();
    

    print snr_value
    return snr_value;

def SNR_Custom(signal, noise):
    #Asignal = (1/signal.shape[0]) * np.sum(np.power(np.abs(signal),2))
    Anoise = (1/noise.shape[0]) * np.sum(np.power(np.abs(noise),2))
    signal_noise = signal+noise
    Atotal = (1/signal_noise.shape[0]) * np.sum(np.power(np.abs(signal_noise),2))
    return 10 * np.log10((Atotal+Anoise)/Anoise)

def l2_norm(x): #assume x is a vector matrix
    return np.sqrt(np.sum(np.abs(x)**2))

def Recovery_Error(original_signal, test_signal):
    return (l2_norm(original_signal - test_signal)/l2_norm(original_signal))

# Data generation using Scikit
###################

# y = Phi * x
# |Phi|_0 = n_nonzero_coefs

#generate sparse signal for later processing using MP or other versions of MP.

'''
y, Phi, x = make_sparse_coded_signal(n_samples=1,
                                   n_components=n_components,
                                   n_features=n_features,
                                   n_nonzero_coefs=n_nonzero_coefs,
                                   random_state=0)
'''


if len(sys.argv) > 1:
	chosen_mp = sys.argv[1]
else:
	print "Please choose an MP to use"
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

n_nonzero_coefs = input("Level of sparsity? ==> ")

#get y from y_mini file
y_file = tb.open_file("y_mini.h5", 'r')
y = y_file.root.data[:]
y_file.close()

#get Phi from Phi_mini file
file = tb.open_file("Phi_result_mini.h5", 'r')
Phi = file.root.data[:]
file.close()
'''
Phi = np.array(Phi)
y_test = np.dot(Phi, x_test)
'''
noise = np.random.normal(0, 10 * 1/25, y.shape[0])
snr_value = SNR_Custom(y,noise)
print snr_value

#test the chosen mp loop
print "================== Result of using {} ==================". format(chosen_mp)
'''
if chosen_mp == "cosamp":
    res, iterations = cosamp(Phi, y, sparsity_chosen)
'''
if chosen_mp == 'omp-scikit':
	mp_process = OrthogonalMatchingPursuit(n_nonzero_coefs = n_nonzero_coefs)
	mp_process.fit(Phi, y)
	x_mp = mp_process.coef_
else:
	x_mp, _, _= mp_process(Phi, y, ncoef=n_nonzero_coefs, verbose=True)
'''
omp_process = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
omp_process.fit(Phi, y_test)
x_mp = omp_process.coef_
'''
#x_mp = res.coef


#x_omp = res


print "Recovery Error between omp result and original y:"
y_tested = np.dot(Phi, x_mp)
R_error = Recovery_Error(y, y_tested)
rms_cur = np.sqrt(np.mean(abs(y - y_tested)**2, axis=None))
#rms = np.mean(abs(x_mp-x_test)**2, axis=None)
print R_error, rms_cur

print "=============================================================================== \n"

print "\noriginal y: "
print y, np.shape(y)

print "\n\nresulting y with {}: ".format(chosen_mp)
print y_tested, np.shape(y_tested)


sys.exit(0)

'''
idx_r, = coef.nonzero()
idn, = new_coef.nonzero()
print idx_r, idn
print np.shape(new_coef[idn]), np.shape(idn)
plt.subplot(312)
plt.xlim(0, n_components)
plt.title("Recovered signal from noise-free measurements")
plt.stem(idx_r, coef[idx_r])

plt.subplot(313)
plt.xlim(0, n_components)
plt.title("Recovered signal from noise-free measurements using custom OMP")
plt.stem(idn, new_coef[idn])


plt.show()
'''
