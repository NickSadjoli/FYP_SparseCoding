import numpy as np
import math
from sklearn.metrics import mean_squared_error
import tables as tb
import sys

def trendline_fit(x,y):
  z = np.polyfit(x, y, 3)
  p = np.poly1d(z)
  return p

def l2_norm(x): #assume x is a vector matrix
    return np.sqrt(np.sum(np.abs(x)**2))

def Recovery_Error(original_signal, test_signal):
    return (l2_norm(original_signal - test_signal)/l2_norm(original_signal))

def RMS(original_signal, test_signal):
  return math.sqrt(mean_squared_error(original_signal,test_signal))
  #return mean_squared_error(original_signal, test_signal)


#function to create a new file to insert Phi, split into 2 types based on type of Phi received
def file_create(f_name, Phi, m, n):

  if type(Phi) is np.ndarray:
      print "One numpy Phi array received"
      write_new_file(f_name, Phi, m, n)

  elif type(Phi) is list:
      print "(Assumed) list of NP Phi arrays received"
      write_new_file(f_name, Phi[0], m, n)

      if len(Phi) <= 1:
          return
      else:
          for i in range(1, len(Phi)):
            insert_array(f_name, Phi[i], i, m, n)
  else: 
      print "Received data doesn't match any compatible types! Exiting!"
      sys.exit(0)


#function to write a completely new file, (or overwritting an old one), and inserting a new Phi (assuemd to be NP array type)
def write_new_file(f_name, Phi, m, n):
  file_name = f_name + '.h5'
  #f = tb.open_file('Phi.h5', 'w')
  f = tb.open_file(file_name, 'w')
  filters = tb.Filters(complevel=5, complib='blosc')

  #because by default numpy (in other scripts) operates using float64, this is necessary
  out = f.create_carray(f.root, 'data', tb.Float64Atom(), shape=(m, n), filters=filters) 

  print "h5 file created, now putting Phi from memory to file..."
    
  step = 1000 #this is the number of rows we calculate each loop (example was using bl)
  #this may not the most efficient value
  #look into buffersize usage in PyTables and adopt the buffersite of the
  #carray accordingly to improve specifically fetching performance
   
  #b = b.tocsc() #we slice b on columns, csc improves performance #not necessary in this case
   
  #this can also be changed to slice on rows instead of columns (which is what will be done for Phi)
  for i in range(0, n, step):
    out[:,i:min(i+step, n)] = Phi[:, i:min(i+step, n)] # initially, example was using this => (a.dot(b[:,i:min(i+bl, l)])).toarray()
    print i
  print "Phi saving done, closing file..."
   
  f.close()


#function to INSERT a new array to a presumably ALREADY EXISTING file!
def insert_array(f_name, array, idx, m, n):
  file_name = f_name + '.h5'
  #f = tb.open_file('Phi.h5', 'w')
  f = tb.open_file(file_name, 'a')
  filters = tb.Filters(complevel=5, complib='blosc')

  array_name = 'data_' + str(idx) 

  #because by default numpy (in other scripts) operates using float64, this is necessary
  out = f.create_carray(f.root, array_name, tb.Float64Atom(), shape=(m, n), filters=filters) 

  print "Inserting next Phi slice into file, index: " + str(idx)
    
  step = 1000 #this is the number of rows we calculate each loop (example was using bl)
  #this may not the most efficient value
  #look into buffersize usage in PyTables and adopt the buffersite of the
  #carray accordingly to improve specifically fetching performance
   
  #b = b.tocsc() #we slice b on columns, csc improves performance #not necessary in this case
   
  #this can also be changed to slice on rows instead of columns (which is what will be done for Phi)
  for i in range(0, n, step):
    out[:,i:min(i+step, n)] = array[:, i:min(i+step, n)] # initially, example was using this => (a.dot(b[:,i:min(i+bl, l)])).toarray()
    print i
  print "Phi saving done, closing file..."
   
  f.close()

def print_sizes(Phi,y):
    print np.shape(Phi), np.shape(y)
    sys.exit(0)


def count_nodes(file):
    count = 0
    for node in file:
      count +=1
    return count -1 #First node is the 'root' group node of the file, which we are not interested in

def print_nodes(file):
    for node in y_file:
        print node[:]
    file.close()
    sys.exit(0)

def take_node_data(file, counter): 
    if counter > 1:
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


    else:
        Phi = file.root.data[:]
    file.close()
    return ar_list
