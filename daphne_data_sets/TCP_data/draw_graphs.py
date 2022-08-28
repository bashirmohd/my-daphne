#draw graphs

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn import decomposition
from sklearn.cluster import KMeans
import os

import pandas as pd 
from sklearn import cluster, datasets


from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA, IncrementalPCA
from pylab import plot, show, savefig, xlim, figure, hold, ylim, legend, boxplot, setp, axes




cfile_normal_data = os.path.join("../cubic/normal/cut_data.csv")
cdfnormaldata=pd.read_csv(cfile_normal_data)

cfile_normal_master=os.path.join("../cubic/normal/cut_master.csv")
cdfnormalmaster=pd.read_csv(cfile_normal_master)



rfile_normal_data = os.path.join("../reno/normal/cut_data.csv")
rdfnormaldata=pd.read_csv(rfile_normal_data)

rfile_normal_master=os.path.join("../reno/normal/cut_master.csv")
rdfnormalmaster=pd.read_csv(rfile_normal_master)



hfile_normal_data = os.path.join("../hamilton/normal/cut_data.csv")
hdfnormaldata=pd.read_csv(hfile_normal_data)

hfile_normal_master=os.path.join("../hamilton/normal/cut_master.csv")
hdfnormalmaster=pd.read_csv(hfile_normal_master)




cfile_dup1_data = os.path.join("../cubic/packet_dup_1/cut_data.csv")
cdfdup1data=pd.read_csv(cfile_dup1_data)

#file_dup1_master=os.path.join("../cubic/packet_dup_1/master_original.csv")
#dfdup1master=pd.read_csv(file_dup1_master)



rfile_dup1_data = os.path.join("../reno/packet_dup_1/cut_data.csv")
rdfdup1data=pd.read_csv(rfile_dup1_data)



hfile_dup1_data = os.path.join("../hamilton/packet_dup_1/cut_data.csv")
hdfdup1data=pd.read_csv(hfile_dup1_data)




cfile_dup1_master = os.path.join("../cubic/packet_dup_1/cut_master.csv")
cdfdup1master=pd.read_csv(cfile_dup1_master)


rfile_dup1_master = os.path.join("../reno/packet_dup_1/cut_master.csv")
rdfdup1master=pd.read_csv(rfile_dup1_master)

hfile_dup1_master = os.path.join("../hamilton/packet_dup_1/cut_master.csv")
hdfdup1master=pd.read_csv(hfile_dup1_master)




cfile_dup5_data = os.path.join("../cubic/packet_dup_5/cut_data.csv")
cdfdup5data=pd.read_csv(cfile_dup5_data)


rfile_dup5_data = os.path.join("../reno/packet_dup_5/cut_data.csv")
rdfdup5data=pd.read_csv(rfile_dup5_data)


hfile_dup5_data = os.path.join("../hamilton/packet_dup_5/cut_data.csv")
hdfdup5data=pd.read_csv(hfile_dup5_data)


#file_dup5_master=os.path.join("../cubic/packet_dup_5/master_original.csv")
#dfdup5master=pd.read_csv(file_dup5_master)




cfile_dup5_master = os.path.join("../cubic/packet_dup_5/cut_master.csv")
cdfdup5master=pd.read_csv(cfile_dup5_master)


rfile_dup5_master = os.path.join("../reno/packet_dup_5/cut_master.csv")
rdfdup5master=pd.read_csv(rfile_dup5_master)


hfile_dup5_master = os.path.join("../hamilton/packet_dup_5/cut_master.csv")
hdfdup5master=pd.read_csv(hfile_dup5_master)



cfile_loss1_data = os.path.join("../cubic/packet_loss_1/cut_data.csv")
cdfloss1data=pd.read_csv(cfile_loss1_data)

rfile_loss1_data = os.path.join("../reno/packet_loss_1/cut_data.csv")
rdfloss1data=pd.read_csv(rfile_loss1_data)

hfile_loss1_data = os.path.join("../hamilton/packet_loss_1/cut_data.csv")
hdfloss1data=pd.read_csv(hfile_loss1_data)

#file_loss1_master=os.path.join("../cubic/packet_loss_1/master_original.csv")
#dfloss1master=pd.read_csv(file_loss1_master)


cfile_loss3_data = os.path.join("../cubic/packet_loss_3/cut_data.csv")
cdfloss3data=pd.read_csv(cfile_loss3_data)


rfile_loss3_data = os.path.join("../reno/packet_loss_3/cut_data.csv")
rdfloss3data=pd.read_csv(rfile_loss3_data)



hfile_loss3_data = os.path.join("../hamilton/packet_loss_3/cut_data.csv")
hdfloss3data=pd.read_csv(hfile_loss3_data)


#file_loss3_master=os.path.join("../cubic/packet_loss_3/master_original.csv")
#dfloss3master=pd.read_csv(file_loss3_master)

cfile_loss5_data = os.path.join("../cubic/packet_loss_5/cut_data.csv")
cdfloss5data=pd.read_csv(cfile_loss5_data)

rfile_loss5_data = os.path.join("../reno/packet_loss_5/cut_data.csv")
rdfloss5data=pd.read_csv(rfile_loss5_data)
hfile_loss5_data = os.path.join("../hamilton/packet_loss_5/cut_data.csv")
hdfloss5data=pd.read_csv(hfile_loss5_data)

#file_loss5_master=os.path.join("../cubic/packet_loss_5/master_original.csv")
#dfloss5master=pd.read_csv(file_loss5_master)


cfile_reord2550_data = os.path.join("../cubic/packet_reorder_25_50/cut_data.csv")
cdfreord2550data=pd.read_csv(cfile_reord2550_data)



rfile_reord2550_data = os.path.join("../reno/packet_reorder_25_50/cut_data.csv")
rdfreord2550data=pd.read_csv(rfile_reord2550_data)



hfile_reord2550_data = os.path.join("../hamilton/packet_reorder_25_50/cut_data.csv")
hdfreord2550data=pd.read_csv(hfile_reord2550_data)



#file_reord2550_master=os.path.join("../cubic/packet_reorder_25_50/master_original.csv")
#dfreord2550master=pd.read_csv(file_reord2550_master)

cfile_reord2550_master= os.path.join("../cubic/packet_reorder_25_50/cut_master.csv")
cdfreord2550master=pd.read_csv(cfile_reord2550_master)



rfile_reord2550_master = os.path.join("../reno/packet_reorder_25_50/cut_master.csv")
rdfreord2550master=pd.read_csv(rfile_reord2550_master)



hfile_reord2550_master = os.path.join("../hamilton/packet_reorder_25_50/cut_master.csv")
hdfreord2550master=pd.read_csv(hfile_reord2550_master)



cfile_reord5050_data = os.path.join("../cubic/packet_reorder_50_50/cut_data.csv")
cdfreord5050data=pd.read_csv(cfile_reord5050_data)

rfile_reord5050_data = os.path.join("../reno/packet_reorder_50_50/cut_data.csv")
rdfreord5050data=pd.read_csv(rfile_reord5050_data)


hfile_reord5050_data = os.path.join("../hamilton/packet_reorder_50_50/cut_data.csv")
hdfreord5050data=pd.read_csv(hfile_reord5050_data)

#file_reord5050_master=os.path.join("../cubic/packet_reorder_50_50/master_original.csv")
#dfreord5050master=pd.read_csv(file_reord5050_master)
cfile_reord5050_master = os.path.join("../cubic/packet_reorder_50_50/cut_master.csv")
cdfreord5050master=pd.read_csv(cfile_reord5050_master)

rfile_reord5050_master= os.path.join("../reno/packet_reorder_50_50/cut_master.csv")
rdfreord5050master=pd.read_csv(rfile_reord5050_master)


hfile_reord5050_master = os.path.join("../hamilton/packet_reorder_50_50/cut_master.csv")
hdfreord5050master=pd.read_csv(hfile_reord5050_master)



#bpc = plt.boxplot(cdfnormalmaster['ThroughputC2S'], positions=1, sym='', widths=0.6)
#bpr = plt.boxplot(rdfnormalmaster['ThroughputC2S'], positions=2, sym='', widths=0.6)
#bph = plt.boxplot(hdfnormalmaster['ThroughputC2S'], positions=3, sym='', widths=0.6)


#data=[cdfnormalmaster['ThroughputC2S'],rdfnormalmaster['ThroughputC2S'],hdfnormalmaster['ThroughputC2S'] ]
#set_box_color(bpc, '#D7191C') # colors are from http://colorbrewer2.org/
#set_box_color(bpr, '#2C7BB6')
#set_box_color(bph, '#2ca25f')

# draw temporary red and blue lines and use them to create a legend
#plt.plot([], c='#D7191C', label='Cubic')
#plt.plot([], c='#2C7BB6', label='Reno')
#plt.plot([], c='#2ca25f', label='Hamilton')
#plt.legend()



# function for setting the colors of the box plots pairs

data = [cdfnormaldata['s_rtt_avg:52'], cdfreord2550data['s_rtt_avg:52'], cdfreord5050data['s_rtt_avg:52']]

fig4, ax4 = plt.subplots()
#ax4.set_title('ThroughputC2S-Data Node')
ax4.set_ylabel('Average RTT (ms)')
ax4.boxplot(data, showfliers=False)
plt.xticks([1, 2, 3], ['Normal', 'PReorder25-50%', 'PReorder50-50%'])

#'C-Reor25-50%', 'C-Reor50-50%', 'R-Normal', 'R-Reor25-50%', 'R-Reor50-50%','H-Normal', 
#	'H-Reor25-50%', 'H-Reor50-50%'])

plt.show()

