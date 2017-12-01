import sys
import numpy as np
import tables as tb
import threading
#from queue import Queue
import time

from sklearn.linear_model import OrthogonalMatchingPursuit
import matplotlib.pyplot as plt

from mp_functions import *
from utils import *


print_lock = threading.Lock()

x_comp = [None] * 4

def print_sizes(Phi,y):
	print np.shape(Phi), np.shape(y)
	sys.exit(0)

def process_result(slc, x_object, x_index, s_cur):
	global Phi
	#global slice_size
	x_object[x_index] , _, _= mp_process(Phi, slc, ncoef=s_cur, verbose=False)
	'''
	with print_lock:
		print threading.current_thread().name + "Done "
	'''
def process_fit(process, Phi, y):
	process.fit(Phi,y)
	x = process.coef_
	return x

def dot_calc (Phi, x_tst):
	res = np.dot(Phi,x_tst)

	return np.dot(Phi,x_tst)

def process_y(y_input, s_thread):
	t0 = time.time()
	#global Phi
	global slice_size
	threads = [None] * (len(y_input)/slice_size)
	x_test = [None] * (len(y_input)/slice_size)
	sparsity_tot = s_thread * (len(y_input)/slice_size)

	for i in range(len(threads)):
		#threads[i]= threading.Thread(target=mp_process,args=(Phi, y_slice[i*slice_size : (i+1)*slice_size]), kwargs={'ncoef':4, 'verbose': False}) #for positional arguments in omp function
		threads[i]= threading.Thread(target=process_result, args=(y_input[i*slice_size : (i+1)*slice_size], x_test, i, s_thread) ) 
		threads[i].daemon = True
		threads[i].start()
		#threads.append(tr)

	for j in range(len(threads)):
		threads[j].join()
	#tf = time.time()
	#print "time to complete all calculations", tf-t0

	#ti = time.time()
	x_res = None
	for x_comp in x_test:
		#print x_comp
		if x_res is None:
			x_res = [x_comp]
		else:
			x_res = x_res + [x_comp]
	td = time.time()
	#print "time to complete concatenation: ", td - ti
	return x_res, sparsity_tot, td-t0


file = tb.open_file("Phi_result.h5", 'r')
Phi = file.root.data[:]
file.close()
y_file = tb.open_file("y_mini.h5", 'r')
y = y_file.root.data[:]
y_file.close()	
step = 256	
#y = np.random.normal(0,30,[step*7,1])
slice_size = Phi.shape[0]
#print_sizes(Phi, y)
sparsity = 20
y_list = []
x_list = []
sparsity_list = []
runtime_list = []
error_list = []
tprev = 0

chosen_mp = 'omp'

mp_process = omp

#print "testing time: ", time.time()

print "time elapsed for each y_slice: "
for i in range(0, len(y), step):
	y_cur = y[i:i+step]
	print 'slice '+ str(i/step)
	cur_s = ((i/step)+1)*5

	if cur_s == 35:
		cur_s = 40

	t0 = time.time()
	#num_of_x = len(y_test)/slice_size
	
	'''
	omp_process = OrthogonalMatchingPursuit(n_nonzero_coefs=1000)
	x_test1 = process_fit(omp_process, Phi, y_test[0:slice_size])
	x_test2 = process_fit(omp_process, Phi, y_test[slice_size: slice_size*2])
	x_test3 = process_fit(omp_process, Phi, y_test[slice_size*2: slice_size*3])
	x_test4 = process_fit(omp_process, Phi, y_test[slice_size*3: slice_size*4])
	'''
	
	'''
	x_test1, _, _ = mp_process(Phi, y_test[0:slice_size], ncoef=100, verbose=False)
	x_test2, _, _ = mp_process(Phi, y_test[slice_size: slice_size*2], ncoef=100, verbose=False)
	x_test3, _, _ = mp_process(Phi, y_test[slice_size*2: slice_size*3], ncoef=1000, verbose = False)
	x_test4, _, _ = mp_process(Phi, y_test[slice_size*3: slice_size*4], ncoef=1000, verbose=False)
	
	#x_test, _, _ = mp_process(Phi, y_test, 'omp', ncoef=20, verbose=False)
	x_test = [x_test1]+[x_test2] +[x_test3]+ [x_test4]
	'''
	x_test, sparsity, mp_time = process_y(y_cur, cur_s)
	#tf = time.time()
	y_list.append(y_cur)
	x_list.append(x_test)
	sparsity_list.append(sparsity)
	runtime_list.append(mp_time)
	tf = time.time()
	print "total time: " + str(tf-t0) + ' ' + str(cur_s)
	print "" 

#print len(y_list), len(x_list)
print "Errors experienced by each y slice: "
for j in range(0, len(y_list)):
	#print x_list[j]
	#print y_list[j]
	y_test = None
	#print x_list[j][0]
	for k in range(len(x_list[j])):
		#print x_test
		if y_test is None:
			y_test = np.dot(Phi, x_list[j][k])
		else:
			y_test = np.concatenate((y_test, np.dot(Phi, x_list[j][k])))
	
	#y_test = np.concatenate( ( np.dot(Phi,x_list[j][0]), np.dot(Phi,x_list[j][1]), np.dot(Phi,x_list[j][2]), np.dot(Phi, x_list[j][3]) ) )
	#y_test = np.dot(Phi, x_list[j])
	#print np.shape(np.dot(Phi, x_list[j][0]))
	y_test = np.reshape(y_test, (len(y_test), 1))
	#print np.shape(y_test)
	#print np.shape(y_list[j]), np.shape(y_test)
	#rms = np.sqrt(np.mean(abs(y_list[j] - y_test)**2, axis=None))
	rms = RMS(y_list[j], y_test)
	r_error = Recovery_Error(y_list[j],y_test)
	#print Recovery_Error(y_list[j],y_test) , rms
	print r_error, rms
	#error_list.append(Recovery_Error(y_list[j],y_test))
	error_list.append(r_error)
#print np.shape(yprint)


#print and runtime relative to sparsity values
''' 
plt.figure(1)
ax = plt.subplot2grid((2,1), (0,0))
plt.plot(sparsity_list, error_list, 'go')
error_trend = trendline_fit(sparsity_list, error_list)
plt.plot(sparsity_list, error_trend(sparsity_list), 'g--')
plt.ylabel('Recovery_Error of {}'.format(chosen_mp))
plt.xlabel('Sparsity')

plt.subplot2grid((2,1), (1,0))
ax = plt.plot(sparsity_list, runtime_list, 'ro')
for xy in zip(sparsity_list, runtime_list):
	ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
runtime_trend = trendline_fit(sparsity_list, runtime_list)
plt.plot(sparsity_list, runtime_trend(sparsity_list), 'r--')
plt.ylabel('Runtime w/ stable noise for {} '.format(chosen_mp))
plt.xlabel('Sparsity')len
'''

f, axgrid = plt.subplots(3, sharex=True)
axgrid[0].plot(sparsity_list, error_list, 'go')
error_trend = trendline_fit(sparsity_list, error_list)
axgrid[0].plot(sparsity_list, error_trend(sparsity_list), 'g--')
axgrid[0].set_title('Recovery_Error of {} vs Sparsity'.format(chosen_mp))
#axgrid[0].xlabel('Sparsity')
'''
axgrid[1].plot(sparsity_list, runtime_list, 'ro')
for i, j in zip(sparsity_list, runtime_list):
	if i <= 4000:
		axgrid[1].annotate('(%i, %.2f)' % (i,j), xy=(i-2, j+0.5), textcoords='data')
	if i > 4000:
		axgrid[1].annotate('(%i, %.2f)' % (i,j), xy=(i-2, j-0.5), textcoords='data')
runtime_trend = trendline_fit(sparsity_list, runtime_list)
axgrid[1].plot(sparsity_list, runtime_trend(sparsity_list), 'r--')
axgrid[1].set_title('Runtime for {} vs Sparsity'.format(chosen_mp))
'''
#print y_list[0]
ori_y = np.sort(y_list[0], kind='heapsort')[-255:]
#ori_y = y_list[ori_y_index]
y_tst = None
for k in range(len(x_list[0])):
		#print x_test
		if y_tst is None:
			y_tst = np.dot(Phi, x_list[0][k])
		else:
			y_tst = np.concatenate((y_tst, np.dot(Phi, x_list[0][k])))
y_tst = np.reshape(y_tst, (len(y_tst),1))
y_tst = np.sort(y_tst, kind='heapsort')[-255:]
#y_tst = y_tst[y_tst_index]


axgrid[1].plot(range(len(ori_y)),ori_y, 'ro')
axgrid[1].set_title('Original_y vs tested_y')
axgrid[1].plot(range(len(y_tst)),y_tst, 'bo')
#axgrid[2].set_title('Tested_y')
'''
for i, j in zip(sparsity_list, runtime_list):
	if i <= 4000:
'''
#axgrid[1].xlabel('Sparsity')


plt.show()

sys.exit(0)

'''
plt.subplot2grid((3,0), (2,0), colspan=1)
plt.plot(sparsity_values, R_error, 'yo')
R_error_trend = trendline_fit(sparsity_values, R_error)
plt.plot(sparsity_values, R_error_trend(sparsity_values), 'y--')
plt.ylabel('RE for {} '.format(chosen_mp))
plt.xlabel('Sparsity')
'''







