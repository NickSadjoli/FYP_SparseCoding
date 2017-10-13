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
from Phi import Phi
import matplotlib.pyplot as plt
import unittest
import math

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


if len(sys.argv) > 1:
	chosen_mp = sys.argv[1]
else:
	print "Please choose an MP to use"
	sys.exit(0)

n_nonzero_coefs = input("Level of sparsity? ==> ")
n_components = 128
n_features = int(2.5*n_nonzero_coefs*math.log10(n_components))
#n_nonzero_coefs = input("Level of sparsity? ==> ")

# generate the data
###################

# y = Phi * x
# |Phi|_0 = n_nonzero_coefs

#generate sparse signal for later processing using MP or other versions of MP.

x = [ 7.14872976, 7.99683485, 6.77462035, 7.35682238, 3.46190329, 8.55134831, 1.23216904, 8.02565473, \
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
y, Phi, x = make_sparse_coded_signal(n_samples=1,
                                   n_components=n_components,
                                   n_features=n_features,
                                   n_nonzero_coefs=n_nonzero_coefs,
                                   random_state=0)
'''
x_actual = np.array(x)
x_test = np.zeros(x_actual.shape[0])
x_test[0:n_nonzero_coefs] = x_actual[0:n_nonzero_coefs]

n = len(x_test)
m = 105 #int(2.5*n_nonzero_coefs*math.log10(n))
#Phi = np.random.normal(0, 0.5, [m,n])
Phi = np.array(Phi)
y_test = np.dot(Phi, x_test)
noise = np.random.normal(0, 10 * 1/25, y_test.shape[0])
snr_value = SNR_Custom(y_test,noise)
print snr_value

#test the chosen mp loop
print "================== Result of using {} ==================". format(chosen_mp)
'''
if chosen_mp == "cosamp":
    res, iterations = cosamp(Phi, y, sparsity_chosen)
'''

x_mp, num_of_it, _= mp_process(Phi, y_test, chosen_mp, ncoef=n_nonzero_coefs, verbose=True)
'''
omp_process = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
omp_process.fit(Phi, y_test)
x_mp = omp_process.coef_
'''
#x_mp = res.coef


#x_omp = res


print "Recovery Error between omp result and original x:"
R_error = Recovery_Error(x_test, x_mp)
rms_cur = np.sqrt(np.mean(abs(x_test - x_mp)**2, axis=None))
#rms = np.mean(abs(x_mp-x_test)**2, axis=None)
print R_error, rms_cur

print "=============================================================================== \n"

print "\noriginal x: "
print x_test

print "\n\nresulting x with {}: ".format(chosen_mp)
print x_mp

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
