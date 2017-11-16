import numpy as np
import tables as tb
import threading
import time

from sklearn.linear_model import OrthogonalMatchingPursuit

from mp_functions import *
from utils import *

def process_fit(process, Phi, y):
	process.fit(Phi,y)
	x = process.coef_
	return x

def dot_calc (Phi, x_tst):
	res = np.dot(Phi,x_tst)

	return np.dot(Phi,x_tst)

file = tb.open_file("Phi.h5", 'r')
Phi = file.root.data[:]
file.close()
step = 25000
y = np.random.normal(0,30,[step*4,1])
slice_size = Phi.shape[0]
sparsity = 20
y_list = []
x_list = []
tprev = 0

'''
def y_injector():
    global y
    global tprev
    threading.Timer(1,y_injector).start() 
    t0 = time.time()
    y = np.random.normal(0,30,[25000,1])
    tf = time.time()
    #print y[0], tf-t0, tf-tprev
    tprev = tf
    #time.sleep(1000)

#y_injector()
injector = threading.Timer(1, y_injector)
injector.start()
'''
print "time elapsed for each y_slice: "
for i in range(0, len(y), step):
	y_test = y[i:i+step]
	t0 = time.time()
	#num_of_x = len(y_test)/slice_size
	
	'''
	omp_process = OrthogonalMatchingPursuit(n_nonzero_coefs=1000)
	x_test1 = process_fit(omp_process, Phi, y_test[0:slice_size])
	x_test2 = process_fit(omp_process, Phi, y_test[slice_size: slice_size*2])
	x_test3 = process_fit(omp_process, Phi, y_test[slice_size*2: slice_size*3])
	x_test4 = process_fit(omp_process, Phi, y_test[slice_size*3: slice_size*4])
	'''
	
	
	x_test1, _, _ = mp_process(Phi, y_test[0:slice_size], 'omp', ncoef=4, verbose=False)
	x_test2, _, _ = mp_process(Phi, y_test[slice_size: slice_size*2], 'omp', ncoef=4, verbose=False)
	x_test3, _, _ = mp_process(Phi, y_test[slice_size*2: slice_size*3], 'omp', ncoef=4, verbose = False)
	x_test4, _, _ = mp_process(Phi, y_test[slice_size*3: slice_size*4], 'omp', ncoef=4, verbose=False)
	
	#x_test, _, _ = mp_process(Phi, y_test, 'omp', ncoef=20, verbose=False)
	x_test = [x_test1]+[x_test2]+[x_test3]+ [x_test4]
	#tf = time.time()
	y_list.append(y_test)
	x_list.append(x_test)
	tf = time.time()
	print tf-t0

#print len(y_list), len(x_list)
print "Errors experienced by each y slice: "
for j in range(0, len(y_list)):
	y_test = np.concatenate((np.dot(Phi,x_list[j][0]), np.dot(Phi,x_list[j][1]), np.dot(Phi,x_list[j][2]), np.dot(Phi, x_list[j][3])))
	#y_test = np.dot(Phi, x_list[j])
	#print np.shape(np.dot(Phi, x_list[j][0]))
	y_test = np.reshape(y_test, (len(y_test), 1))
	#print np.shape(y_list[j]), np.shape(y_test)
	print Recovery_Error(y_list[j],y_test)
#print np.shape(yprint)


#for j in range(0,len(x_list)):







