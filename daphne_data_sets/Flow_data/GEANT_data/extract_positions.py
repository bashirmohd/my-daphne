import xmltodict
import numpy as np
import pprint
import glob
from collections import OrderedDict

import sys


START = 3900 
END = 3967

# SPecifying the range of values that can be used
if len(sys.argv) > 1: 
  if sys.argv[1]:
    START=int(sys.argv[1])
  elif sys.argv[2]:
    END=int(sys.argv[2])
  print ("The script has the START:"+sys.argv[0]+"END:"+ sys.argv[1])



def create_matrix_data(file_name):
  # for index, p in enumerate(file_name):
  #   # print(index)
  print(file_name)
  with open(file_name) as fd:
      doc = xmltodict.parse(fd.read())
  print(doc)
    # for i in range(0,len(doc[u'TrafficMatrixFile'][u'IntraTM']['src'])):
    #   id = -1  

    #   for i, j in doc[u'TrafficMatrixFile'][u'IntraTM']['src'][i].items():
    #     if i == '@id':
    #       id = int(j)-1
    #     try: 
    #       if isinstance(j, list):
    #           for k in j:
    #             if isinstance(k,OrderedDict):
    #               last_id = -1
    #               # print k
    #               for l,m in k.items():
    #                 # print l,m
    #                 if l == "@id":
    #                   last_id = int(m)-1
    #                 elif l == "#text":
    #                   id_Core[index][int(id)][int(last_id)] = float(m)/100
    #     except IndexError:
    #       pass 
      # fd.close()

Geant_positions = np.zeros((23,2),dtype=np.float) 


file_name = "Geant_positions.xml"

# print("START:", file_name[START],"\nEND:", file_name[END]) 
print("The length of the filename", len(file_name))

create_matrix_data(file_name)

print np.shape(id_Core)

outfile = open('time_varying_positions/time_.npy','w')
np.save(outfile, id_Core)
outfile.close()

print len(id_Core)
