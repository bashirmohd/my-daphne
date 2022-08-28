import xmltodict
import numpy as np
import pprint
import glob
from collections import OrderedDict

import sys


START =4900
END = 4967

THRESHOLD_VALUE=900

# Specifying the range of values that can be used
if len(sys.argv) > 1:
  if sys.argv[1]:
    START=int(sys.argv[1])
  elif sys.argv[2]:
    END=int(sys.argv[2])
  print ("The script has the START:"+sys.argv[0]+"END:"+ sys.argv[1])



def create_matrix_data(file_name):
  for index, p in enumerate(file_name):
    # print(index)
    with open(p) as fd:
        doc = xmltodict.parse(fd.read())
    for i in range(0,len(doc[u'TrafficMatrixFile'][u'IntraTM']['src'])):
      id = -1  

      for i, j in doc[u'TrafficMatrixFile'][u'IntraTM']['src'][i].items():
        if i == '@id':
          id = int(j)-1
        try: 
          if isinstance(j, list):
              for k in j:
                if isinstance(k,OrderedDict):
                  last_id = -1
                  # print k
                  for l,m in k.items():
                    # print l,m
                    if l == "@id":
                      last_id = int(m)-1
                    elif l == "#text":
                      id_Core[index][int(id)][int(last_id)] = float(m)/100
        except IndexError:
          pass 
      fd.close()


id_Core = np.zeros((67,23,23),dtype=np.int) 

# pprint.pprint(glob.glob("traffic-matrices/*.xml"))

file_name = glob.glob("traffic-matrices/*.xml")
file_name = sorted(file_name) 

print("START:", file_name[START],"\nEND:", file_name[END]) 
print("The length of the filename", len(file_name))

create_matrix_data(file_name[START:END])

id_Core[id_Core <= THRESHOLD_VALUE] = 0

print np.shape(id_Core)
print("This is the file location", 'time_varying_data/time_'+str(file_name[START].split('/')[-1])+'_'+str(file_name[END].split('/')[-1])+'.npy')
outfile = open('time_varying_data/time_'+str(file_name[START].split('/')[-1])+'_'+str(file_name[END].split('/')[-1])+'.npy','w')
np.save(outfile, id_Core)
outfile.close()

print len(id_Core)
