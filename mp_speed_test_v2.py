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

def process_result(slc, x_object, x_index, s_cur):
	global Phi
	#global slice_size
	#x_object[x_index] , _, _= mp_process(Phi, slc, ncoef=s_cur, verbose=False)
	x_object[:, x_index], _, _ = mp_process(Phi, slc, ncoef=s_cur, verbose=False)
	'''
	with print_lock:
		print threading.current_thread().name + "Done "
	'''
def process_multi(slc, x_object, x_index, s_cur):
	global Phi
	#global slice_size
	#x_object[x_index] , _, _= mp_process(Phi, slc, ncoef=s_cur, verbose=False)
	x_object[:, x_index], _, _ = mp_process(Phi[x_index], slc, ncoef=s_cur, verbose=False)
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
	#global step
	#print len(y_input)
	#y_cur = np.reshape(y_input, (m, len(y_input)/m))
	y_cur = np.reshape(y_input, (len(y_input)/m, m))
	y_cur = y_cur.T
	k = np.shape(y_cur)[1]

	#do sanity check for amount of Phi available for use for all slices of y in threads later
	if k != counter:
		print "Amount of available Phi doesn't match the amount of y slices formed! Please ensure correct Phi was selected!"
		sys.exit(0)

	#print_sizes(y_cur, y_cur)
	#k = np.shape(y_cur)[0]
	#global slice_size
	threads = [None] * (k)
	#x_test = [None] * (k)
	x_test = np.zeros((n,k))
	
	#print_sizes(x_test[:, 0],y_cur)
	sparsity_tot = s_thread * (k)

	for i in range(len(threads)):
		#threads[i]= threading.Thread(target=process_result, args=(y_input[i*slice_size : (i+1)*slice_size], x_test, i, s_thread) ) 
		#threads[i] = threading.Thread(target=process_result, args=(y_cur[i], x_test, i, s_thread) )
		if counter > 1: #i.e. more than one Phi available
			threads[i] = threading.Thread(target=process_multi, args=(y_cur[:, i], x_test, i, s_thread))
		else:
			threads[i] = threading.Thread(target=process_result, args=(y_cur[:, i], x_test, i, s_thread) )
		threads[i].daemon = True
		threads[i].start()
		#threads.append(tr)

	for j in range(len(threads)):
		threads[j].join()
	#tf = time.time()
	#print "time to complete all calculations", tf-t0

	#ti = time.time()
	#x_res = np.zeros((n,k))
	#print np.shape(x_test)
	#print_sizes(y_cur, x_test)
	#x_test = np.array(x_test)
	#print x_test.shape
	#x_res = x_test.T #transpose the result since the actual returned matrix is supposed to be the other way.
	#print np.shape(x_test)

	#x_res = np.reshape(x_test, (np.shape(x_test)[1], np.shape(x_test)[0] ))
	#print_sizes(y_cur, x_res)
	'''
	for x_comp in x_test:
		#print x_comp
		if x_res is None:
			x_res = [x_comp]
		else:
			x_res = x_res + [x_comp]
	'''
	x_res = x_test
	td = time.time()
	#print "time to complete concatenation: ", td - ti
	return x_res, sparsity_tot, td-t0

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



repeats = 10
#file = tb.open_file("Phi_result_mini_ii.h5", 'r')
file = tb.open_file("Phi_result_working.h5", 'r')
#Phi = file.root.data[:]
counter = count_nodes(file)
print counter
if counter > 1:
	m,n = (0,0)
	Phi = [None] * counter
	c = 0
	for node in file:
		if c!= 0:
			n_index = node.name.lstrip('data_')

			if n_index == '':
				n_index = 0
			else:
				n_index = int(n_index)
			Phi[n_index] = node[:]
		c+=1
	m,n = np.shape(Phi[0])

else:
	Phi = file.root.data[:]
	m,n = np.shape(Phi)

'''
for g in Phi:
	print g
	print np.shape(g), "######"
'''

file.close()


#sys.exit(0)

y_file = tb.open_file("y_large.h5", 'r')
y = y_file.root.data[:]
step = len(y)
#y = np.repeat(y, repeats, axis=0)		
y_file.close()
print len(y)
#m, n = np.shape(Phi)
#step = len(y)
#y = np.random.normal(0,30,[step*7,1])
#slice_size = Phi.shape[0]
#print_sizes(Phi, y)
#sparsity = 20
y_list = [] 
x_list = []
sparsity_list = []
runtime_list = []
error_list = []
rms_list = []
tprev = 0

chosen_mp = 'omp'

mp_process = omp

#print "testing time: ", time.time()

print "time elapsed for each y_slice: "
#for i in range(0, len(y), step):
for i in range(1, repeats):
	#y_cur = y[i:i+step]
	y_cur = y
	#print 'slice '+ str(i/step)
	print 'slice' + str(i)
	#cur_s = ((i/step)+1)* 5
	#cur_s = ((i/step)+1)* 1 #use this for the large Phi one, since this is gonna be done per thread
	cur_s =  i * 1
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
	#print_sizes(Phi, x_list[j])
	y_test = None
	#print np.shape(x_list[j])
	#x_print = np.reshape(x_list[j], (len(x_list[j]), 1))
	#print x_list[j]
	if counter > 1:
		for k in range(0,np.shape(x_list[j])[1]):
			slc = np.dot(Phi[k], x_list[j][:, k])
			if y_test is None:
				y_test = slc
			else:
				y_test = np.concatenate((y_test, slc))
		#y_res = y_test
		#y_res = np.dot(Phi[j], x_list[j])
	else:
		y_test = np.dot(Phi, x_list[j])
		#y_res = y_test
		#y_res = np.dot(Phi, x_list[j])
	#print np.shape(y_res)
	#print np.shape(y_list[j]), np.shape(y_test)
	#rms = np.sqrt(np.mean(abs(y_list[j] - y_test)**2, axis=None))
	#print_sizes(y_list[j], y_res)
	'''
	
	for k in range(np.shape(y_res)[1]):
		#print np.shape(y_res[:,k])
		if y_test is None:
			y_test = y_res[:, k]
		else:
			y_test = np.concatenate((y_test, y_res[:, k]))
			#print np.shape(y_test)
	'''

	'''
	for k in range(np.shape(y_res)[0]):
		if y_test is None:
			y_test = y_res[k]
		else:
			y_test = np.concatenate((y_test, y_res[k]))
	'''
	
	y_test = np.reshape(y_test, (len(y_test), 1) )
	rms = RMS(y_list[j], y_test)
	#rms = ((y_list[j] - y_test) ** 2).mean()
	r_error = Recovery_Error(y_list[j],y_test)
	#print Recovery_Error(y_list[j],y_test) , rms
	print r_error, rms
	#error_list.append(Recovery_Error(y_list[j],y_test))
	error_list.append(r_error)
	rms_list.append(rms)
#print np.shape(yprint)


#print RE, RMS and runtime relative to sparsity values
''' 
plt.figure(1)
ax = plt.subplot2grid((3,1), (0,0))
plt.plot(sparsity_list, error_list, 'go')
error_trend = trendline_fit(sparsity_list, error_list)
plt.plot(sparsity_list, error_trend(sparsity_list), 'g--')
plt.ylabel('Recovery_Error of {}'.format(chosen_mp))
plt.xlabel('Sparsity')

ax = plt.subplot2grid((3,1), (1,0))
plt.plot(sparsity_list, rms_list, 'bo')
error_trend = trendline_fit(sparsity_list, rms_list)
plt.plot(sparsity_list, error_trend(sparsity_list), 'g--')
plt.ylabel('Recovery_Error of {}'.format(chosen_mp))
plt.xlabel('Sparsity')


plt.subplot2grid((3,1), (2,0))
ax = plt.plot(sparsity_list, runtime_list, 'ro')
for xy in zip(sparsity_list, runtime_list):
	ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
runtime_trend = trendline_fit(sparsity_list, runtime_list)
plt.plot(sparsity_list, runtime_trend(sparsity_list), 'r--')
plt.ylabel('Runtime w/ stable noise for {} '.format(chosen_mp))
plt.xlabel('Sparsity')len
'''
'''
for x in x_list:
	print np.shape(x)
	print np.nonzero(x)
'''
f, axgrid = plt.subplots(3)

ind = np.arange(len(sparsity_list))
width = 0.5
rect0 = axgrid[0].bar(ind, error_list, color='r')
axgrid[0].set_ylabel('RE values')
axgrid[0].set_title('Recovery_Error of {} vs Sparsity'.format(chosen_mp))
axgrid[0].set_xticks(ind+width/2)
axgrid[0].set_xticklabels(tuple(sparsity_list))
axgrid[0].legend(error_list, ('RE for different sparsity values'))


rect1 = axgrid[1].bar(ind, rms_list, width, color='b')
axgrid[1].set_ylabel('RMS_Values')
axgrid[1].set_title('RMS for {} vs Sparsity'.format(chosen_mp))
axgrid[1].set_xticks(ind+width/2)
axgrid[1].set_xticklabels(tuple(sparsity_list)) #cannot directly take an array
axgrid[1].legend(rms_list, ('RMS values for each sparsity_values')) 	

rect2 = axgrid[2].bar(ind, runtime_list, width, color='b')
axgrid[2].set_ylabel('Runtimes (s)')
axgrid[2].set_title('Runtimes for {} vs Sparsity'.format(chosen_mp))
axgrid[2].set_xticks(ind+width/2)
axgrid[2].set_xticklabels(tuple(sparsity_list)) #cannot directly take an array
axgrid[2].legend(rms_list, ('Runtime values'))


autolabel(rect0, axgrid[0])
autolabel(rect1, axgrid[1])
autolabel(rect2, axgrid[2])

''' #Non-Barchart version
axgrid[0].plot(sparsity_list, error_list, 'go')
error_trend = trendline_fit(sparsity_list, error_list)
axgrid[0].plot(sparsity_list, error_trend(sparsity_list), 'g--')
axgrid[0].set_title('Recovery_Error of {} vs Sparsity'.format(chosen_mp))

axgrid[1].plot(sparsity_list, rms_list, 'bo')
for i, j in zip(sparsity_list, rms_list):
	if i <= 4000:
		axgrid[1].annotate('(%i)' % (i), xy=(i-2, j+0.5), textcoords='data')
	if i > 4000:
		axgrid[1].annotate('(%i)' % (i), xy=(i-2, j-0.5), textcoords='data')
rms_trend = trendline_fit(sparsity_list, rms_list)
axgrid[1].plot(sparsity_list, rms_trend(sparsity_list), 'b--')
axgrid[1].set_title('RMS for {} vs Sparsity'.format(chosen_mp))


axgrid[2].plot(sparsity_list, runtime_list, 'ro')
for i, j in zip(sparsity_list, runtime_list):
	if i <= 4000:
		axgrid[2].annotate('(%.2f)' % j, xy=(i-2, j+0.5), textcoords='data')
	if i > 4000:
		axgrid[2].annotate('(%.2f)' % j, xy=(i-2, j-0.5), textcoords='data')
runtime_trend = trendline_fit(sparsity_list, runtime_list)
axgrid[2].plot(sparsity_list, runtime_trend(sparsity_list), 'r--')
axgrid[2].set_title('Runtime for {} vs Sparsity'.format(chosen_mp))


'''



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







