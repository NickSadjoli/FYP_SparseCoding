import numpy as np
import sys

if len(sys.argv) == 3:
    try:
        m = int(sys.argv[1])
        n = int(sys.argv[2])
    except ValueError :
        print "Invalid m and n values! Please input m and n as integers only!"
        sys.exit(0)

elif len(sys.argv) == 2 and (sys.argv[1] == '-h' or sys.argv[1] == '-help'):
    print "Random signal generator to be saved and used for testing signal manipulation algorithms. \
            Note that numpy generates random signal each time it is called"
else:
    print "Please input the size parameters of the signal, m and n, in integers!"
    sys.exit(0)
'''
if (type(m) is not int) or (type(n) is not int):
    print type(m),type(n)
    
    sys.exit(0)
'''

print '['
for i in range(0,m):
    row = []
    for k in np.random.normal(0,0.5,[1,n])[0]:
        row.append(k)
    print row ,',' 
print ']'