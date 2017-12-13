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
file = tb.open_file("Phi_result_test.h5", 'r')
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

	x_test, sparsity, mp_time = process_y(y_cur, cur_s)
	#tf = time.time()
	y_list.append(y_cur)
	x_list.append(x_test)
	sparsity_list.append(sparsity)
	runtime_list.append(mp_time)
	tf = time.time()
	print "total time: " + str(tf-t0) + ' ' + str(cur_s)
	print "" 

x_testing = x_list[0] #used for looking at the non-zero elements of

#print len(y_list), len(x_list)
print "Errors experienced by each y slice: "
for j in range(0, len(y_list)):

	y_test = None

	if counter > 1:
		for k in range(0,np.shape(x_list[j])[1]):
			slc = np.dot(Phi[k], x_list[j][:, k])
			if y_test is None:
				y_test = slc
			else:
				y_test = np.concatenate((y_test, slc))

	else:
		y_test = np.dot(Phi, x_list[j])

	
	y_test = np.reshape(y_test, (len(y_test), 1) )
	rms = RMS(y_list[j], y_test)
	#rms = ((y_list[j] - y_test) ** 2).mean()
	r_error = Recovery_Error(y_list[j],y_test)
	#print Recovery_Error(y_list[j],y_test) , rms
	print r_error, rms
	#error_list.append(Recovery_Error(y_list[j],y_test))
	error_list.append(r_error)
	rms_list.append(rms)

#print RE, RMS and runtime relative to sparsity values
print x_testing[0]

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
axgrid[2].legend(runtime_list, ('Runtime values'))

'''
ind_x = np.arange(len(x_testing[0]))
rect3 = axgrid[3].bar(ind_x, x_testing[0], width, color='b')
axgrid[3].set_ylabel('non-zero values')
axgrid[3].set_title('Runtimes for {} vs Sparsity'.format(chosen_mp))
axgrid[3].set_xticks(ind_x+width/2)
axgrid[3].set_xticklabels(tuple(ind_x)) #cannot directly take an array
axgrid[3].legend(x_testing[0], ('Runtime values'))
'''

autolabel(rect0, axgrid[0])
autolabel(rect1, axgrid[1])
autolabel(rect2, axgrid[2])
#autolabel(rect3, axgrid[3])


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









