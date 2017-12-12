'''
Author: Nicholas Sadjoli
Description: Test script for checking performance of chosen MP based on varying sparsity of input signal
'''

from __future__ import division
import sys 
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import signaltonoise
from scipy.signal import argrelextrema
from pylab import * 

from sklearn.preprocessing import normalize
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.datasets import make_sparse_coded_signal
import numpy as np


from mp_functions import * #mp_functions contains all the MP algorithms to be tested.
from utils import *
from Phi import Phi
import matplotlib.pyplot as plt
import unittest

import operator


def SNR_Custom(signal, noise): # definition of SNR = 10 * log10(Psignal/Pnoise), where Psignal = (1/len(signal)) * sum(signal[i]^2)
    #Asignal = (1/signal.shape[0]) * np.sum(np.power(np.abs(signal),2))
    Anoise = (1/noise.shape[0]) * np.sum(np.power(np.abs(noise),2))
    signal_noise = signal+noise
    Atotal = (1/signal_noise.shape[0]) * np.sum(np.power(np.abs(signal_noise),2))
    return 10 * np.log10((Atotal+Anoise)/Anoise)


#used to automatically place labels on the rectangular bar chart below
def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.2f' % height,
                ha='center', va='bottom')


chosen_mp = None
max_iter = None
mp_process = None

if len(sys.argv) > 1:
  if len(sys.argv) == 2:
    chosen_mp = sys.argv[1]
  elif len(sys.argv) == 3:
    chosen_mp = sys.argv[1]
    max_iter = sys.argv[2]
else:
	print "Please choose an MP to use"
	sys.exit(0)

if chosen_mp == 'omp':
  mp_process = omp
elif chosen_mp == 'bomp':
  mp_process = bomp
elif chosen_mp == 'cosamp':
  mp_process = cosamp
elif chosen_mp == 'bmpnils':
  mp_process = bmpnils


rms = []
R_error = []
rms_w_noise = []
R_error_w_noise = []
runtime = []
runtime_w_noise = []
snr_values = []
sparsity_values = range(1, 81)
'''
if verbose == None:
  verbose = False
else:
  verbose = verbose
'''

vbose = input("Verbose? ==> ")
'''
if vbose == "yes" or vbose == "Y" or vbose == "YES" or vbose == "y":
  vbose = True
elif vbose == "no" or vbose == "N" or vbose == "NO" or vbose == "n":
  vbose = False
else:
  vbose = False
'''
#get y from y_mini file
y_file = tb.open_file("y_mini.h5", 'r')
y_actual = y_file.root.data[:]
y_file.close()

#get Phi from Phi_mini file
file = tb.open_file("Phi_result_mini.h5", 'r')
Phi = file.root.data[:]
file.close()

#print_sizes(Phi, y_actual)

for s in sparsity_values[:]: #cannot start with sparsity 0, since that means there is no non-zero components anyways
  '''
  # generate the data
  ###################

  # y = Phi * x
  # |Phi|_0 = n_nonzero_coefs

  #generate sparse signal for later processing using MP or other versions of MP.
  '''
  #x_test = x_actual

  '''
  x_test = np.zeros(_actual.shape[0])
  x_test[0:s] = x_actual[0:s]
  
  n = len(x_test)
  m = 105 #int(2.5*s*math.log10(n))

  #Phi = np.random.normal(0, 0.5, [m,n])
  
  y_actual = np.dot(Phi, x_test)
  ''' 

  #y, Phi, x = make_sparse_coded_signal(n_samples=1, n_components=n_components, n_features=n_features, n_nonzero_coefs=s, random_state=0)
  noise = np.random.normal(0, 10 * 1/25, y_actual.shape[0])
  noise = np.reshape(noise, (len(noise),1) )
  #print np.shape(y_actual + noise)

  if chosen_mp == "omp-scikit":
    omp_process = OrthogonalMatchingPursuit(n_nonzero_coefs=s)
    omp_process.fit(Phi, y_actual)
    x_mp = omp_process.coef_
    omp_process_noise = OrthogonalMatchingPursuit(n_nonzero_coefs=s)
    omp_process_noise.fit(Phi, y_actual+noise)
    x_mp_noise = omp_process_noise.coef_
  
  #print verbose
  else:
    
    if max_iter is None:
      x_mp, numit, time = mp_process(Phi, y_actual, ncoef=s, verbose=vbose)
      x_mp_noise,numit,time_noise = mp_process(Phi, y_actual + noise, ncoef=s, verbose=vbose)
    else:
      x_mp, numit, time = mp_process(Phi, y_actual, ncoef=s, maxit=max_iter, verbose=vbose)
      x_mp_noise,numit,time_noise = mp_process(Phi, y_actual + noise, ncoef=s, maxit=max_iter, verbose=vbose)
  
  y_tested = np.dot(Phi, x_mp)
  y_tested = np.reshape(y_tested, (len(y_tested), 1) )
  y_noise = np.dot(Phi, x_mp_noise)
  y_noise = np.reshape(y_noise, (len(y_noise), 1) ) 
  #noise = np.random.normal(0, 1/25, y.shape[0])
  #rms_cur = np.sqrt(( np.sum((y_actual - y_tested) ** 2))/len(y_actual )) #make sure the y_tested and y_actual has exact same size!
  rms_cur = RMS(y_actual, y_tested)
  R_error_cur = Recovery_Error(y_actual, y_tested)
  #rms_noise_cur = np.sqrt(np.mean(abs(y_actual - y_noise)**2, axis=None))
  rms_noise_cur = RMS(y_actual, y_noise)
  R_error_noise_cur = Recovery_Error(y_actual, y_noise)
  
  if chosen_mp == 'bomp':
    if rms_cur < 20:
      rms.append(rms_cur)
    else:
      test, _, _ = mp_process(Phi, y_actual, ncoef=s, verbose=True)
      rms.append(0.5)
    #runtime.append(time)
  
    if rms_noise_cur < 20:
      rms_w_noise.append(rms_noise_cur)
    else:
      rms_w_noise.append(0.5)
    if R_error_cur < 20:
      R_error.append(R_error_cur)
    else:
      R_error.append(0.5)
    if R_error_noise_cur < 20:
      R_error_w_noise.append(R_error_noise_cur)
    else:
      R_error_w_noise.append(0.5)
  
  else:
    rms.append(rms_cur)
    rms_w_noise.append(rms_noise_cur)
    R_error.append(R_error_cur)
    R_error_w_noise.append(R_error_noise_cur)
  
  if chosen_mp != "omp-scikit":
    runtime.append(time)
    runtime_w_noise.append(time_noise)
  snr_values.append(SNR_Custom(y_actual, noise)) 
  print 'done for s = ' + str(s) + ": " + str(rms_cur)

#print snr_values
plt.figure(1)
plt.subplot2grid((4,2), (0,0), colspan=1)
plt.plot(sparsity_values, rms, 'go')
rms_trend = trendline_fit(sparsity_values, rms)
plt.plot(sparsity_values, rms_trend(sparsity_values), 'g--')
plt.ylabel('RMS of {}'.format(chosen_mp))
plt.xlabel('Sparsity')

plt.subplot2grid((4,2), (1,0), colspan=1)
plt.plot(sparsity_values, rms_w_noise, 'ro')
rms_noise_trend = trendline_fit(sparsity_values, rms_w_noise)
plt.plot(sparsity_values, rms_noise_trend(sparsity_values), 'r--')
plt.ylabel('RMS w/ stable noise for {} '.format(chosen_mp))
plt.xlabel('Sparsity')

plt.subplot2grid((4,2), (2,0), colspan=1)
plt.plot(sparsity_values, R_error, 'yo')
R_error_trend = trendline_fit(sparsity_values, R_error)
plt.plot(sparsity_values, R_error_trend(sparsity_values), 'y--')
plt.ylabel('RE for {} '.format(chosen_mp))
plt.xlabel('Sparsity')

plt.subplot2grid((4,2), (3,0), colspan=1)
plt.plot(sparsity_values, R_error_w_noise, 'bo')
R_error_noise_trend = trendline_fit(sparsity_values, R_error_w_noise)
plt.plot(sparsity_values, R_error_noise_trend(sparsity_values), 'b--')
plt.ylabel('RE w/ stable noise for {} '.format(chosen_mp))
plt.xlabel('Sparsity')


plt.subplot2grid((4,2), (0,1), colspan=1)
plt.plot(sparsity_values, snr_values, 'b-')
plt.ylabel('SNR(in dB)')
plt.xlabel('Sparsity')

if chosen_mp != "omp-scikit":

  plt.subplot2grid((4,2), (1,1), colspan=1)
  plt.plot(sparsity_values, runtime, 'ko')
  runtime_trend = trendline_fit(sparsity_values, runtime)
  plt.plot(sparsity_values, runtime_trend(sparsity_values), 'k--')
  plt.ylabel('Runtime of {}'.format(chosen_mp))
  plt.xlabel('Sparsity')
  
  plt.subplot2grid((4,2), (2,1), colspan=1)
  plt.plot(sparsity_values, runtime_w_noise, 'ro')
  runtime_noise_trend = trendline_fit(sparsity_values, runtime_w_noise)
  plt.plot(sparsity_values, runtime_noise_trend(sparsity_values), 'r--')
  plt.ylabel('Runtime w/stable noise of {}'.format(chosen_mp))
  plt.xlabel('Sparsity')

plt.show()

ind = np.arange(len(y_actual))
width = 0.35
rect0 = axgrid[0].bar(ind, y_actual, color='g')
axgrid[0].set_ylabel('Value')
axgrid[0].set_title('Tested y')
axgrid[0].set_xticks(ind+width/2)
axgrid[0].set_xticklabels(tuple(ind))
'''
ind_i = np.arange(len(snr_values))
rect1 = axgrid[1].bar(ind_i, snr_values, color='b')
axgrid[1].set_ylabel('Value')
axgrid[1].set_title('SNR values')
axgrid[1].set_xticks(ind+width/2)
axgrid[1].set_xticklabels(tuple(ind))
'''
ind_i = np.arange(len(noise_actual))
rect1 = axgrid[1].bar(ind_i, noise_actual, color='b')
axgrid[1].set_ylabel('Value')
axgrid[1].set_title('SNR values')
axgrid[1].set_xticks(ind_i+width/2)
axgrid[1].set_xticklabels(tuple(ind_i))


autolabel(rect0, axgrid[0])
autolabel(rect0, axgrid[1])


sys.exit(0)

