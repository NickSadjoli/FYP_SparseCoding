import numpy as np
import math
from sklearn.metrics import mean_squared_error
import tables as tb

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

def file_create(f_name, Phi, m, n):
  print type(f_name)
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
