import numpy as np
print '['
for i in range(0,105):
	row = []
	for k in np.random.normal(0,0.5,[1,128])[0]:
		row.append(k)
	print row ,',' 
print ']'