'''
time it takes to download
bandwidth
file size

# Usage:
# on host_A: On your terminal run this command with the following arguements
#'python <scriptname e.g netpreflight.py> <link to download image from> <number of iterations>' 

# on host_B: No action is required on host_B

'''
import sys, time, wget
from datetime import datetime as dt
from socket import *

host, tests = sys.argv[1], sys.argv[2]
BUFSIZE = 1024
iterations = []

for i in range(int(tests)):
    print('Testing for iteration {}....'.format(i))
    start = dt.now()
    wget.download(host)
    end = dt.now()
    elasped = (end - start).microseconds /1000
    throughtput = round((BUFSIZE*int(tests)*0.001) / (elasped+0.000001), 3)
    print('Elasped {} microseconds'.format(elasped))
    with open('results.txt', 'a') as ff:
        ff.write('iteration {}, {}, {}\n'.format(i, elasped, throughtput))
    ff.close()
    iterations.append(elasped)
print(iterations)


