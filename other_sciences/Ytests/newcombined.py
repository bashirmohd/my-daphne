#combined.py


#Exp1) Gamma ID, number of points detected, Maximum energy radiated, minimum energy radiated, aspect ratio of path taken. 
#Then Ill try to see if we identify points who cluster, to see if they all belong to the same area.

#Exp2) gamma id, energy,x,y,z (as it is now)
#and see what clusters come out

# work out the centroid in each ray PCA


from __future__ import print_function

from sklearn import cluster, datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd 
import os


from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA, IncrementalPCA


label_file = os.path.join("withpartial_full.csv")

#raw_data={'Average rtt C2S', 'Average rtt S2C','target'}
#df=pd.DataFrame(raw_data, columns = ['Sent','Received','Lost','Duplicated','Reordered'])

df=pd.read_csv(label_file)


# load dataset into Pandas DataFrame
df = pd.read_csv(label_file, names=['target','medEnergy', 'xval','yval','zval','pts','maxe','mine'])

print("*******")
print(df)

#extracting features
features = ['medEnergy','xval','yval','zval' ,'pts','maxe','mine']#,'xval','yval','zval']



# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values

uniquetarget=[]
inside=[]

for val in y:
	if val not in uniquetarget:
		uniquetarget.append(val)

for s in range(len(uniquetarget)):
	#inside=np.array(uniquetarget)[0][0]
	print(np.array(uniquetarget)[s][0])
#	inside.append((np.array(uniquetarget)[s][0]))

#print(uniquetarget)
#print(inside)
# Standardizing the features
x = StandardScaler().fit_transform(x)

#print "1"

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

#log = open("pcaresultsfile.csv", "w")

#principalComponents[0:2]

#for x in principalComponents:
#	print("x =")
#	print(x)
#	cellvalue = np.array2string(x, precision=4, separator=',',suppress_small=True)
 #	cellvalue = cellvalue.replace("[", "")
 #	cellvalue = cellvalue.replace("]", "")
 #	print(cellvalue, file = log)

#print(principalDf.size) #print into a csv file

#plot PCA

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111, projection='3d') 
ax.set_xlabel('Principal Component 1', fontsize = 10)
ax.set_ylabel('Principal Component 2', fontsize = 10)
ax.set_ylabel('Principal Component 3', fontsize = 10)

ax.set_title('3 component PCA', fontsize = 10)
targets = ['partial ','full ']
print(targets)
#['Normal', 'Loss1%', 'Loss3%', 'Loss5%', 'Dup1%', 'Dup3%','Dup5%','Reord10%','Reord30%','Reord50%']
#print "2"

colors = ['r','b', 
'black','black','black','black','black','black','black','black','black','black',
'lime','lime','lime','lime','lime','lime','lime','lime','lime','lime',
'yellow', 'yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow',
'yellow', 'cyan', 'coral', 'gray', 'indigo']
for color,target in zip(colors,targets):
	#print(finalDf['target'].size)
	#print(target.size) 
	indicesToKeep = finalDf['target'] == target
	ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
  	, finalDf.loc[indicesToKeep, 'principal component 2']
  	, finalDf.loc[indicesToKeep, 'principal component 3']
  	, c = color
  	, s = 50)
ax.legend(targets)
ax.grid()
plt.show()

print("explained_variance_")
print(pca.explained_variance_)  
#print("singular values")
#print(pca.singular_values_) 

print("variance ratio")
print(pca.explained_variance_ratio_)

print("mean")
print(pca.mean_)



#print "PCA has been conducted, with 2 component as a parameter and we fit the data"

#print "now veiw the new features, number of rows and 2 features" 
#print(principalComponents.shape)

#print "new feature data"
#print(principalComponents)

#print "columns"
#intermediary=pd.DataFrame(pca.components_, columns=list(features)).values
#print(intermediary.reshape(2,26).tolist())


# n_components = 2
# ipca = IncrementalPCA(n_components=n_components, batch_size=10)
# X_ipca = ipca.fit_transform(x)

# pca = PCA(n_components=n_components)
# X_pca = pca.fit_transform(x)

# colors = ['r', 'g', 'b', 'black', 'lime', 'yellow', 'cyan', 'coral']

# for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:
#     plt.figure(figsize=(8, 8))
#     for color, i, target_name in zip(colors, [0, 1], targets):
#         plt.scatter(X_transformed[y == i, 0], X_transformed[y == i, 1],
#                     color=color, lw=2, label=target_name)

#     if "Incremental" in title:
#         err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
#         plt.title(title + " of tstat dataset\nMean absolute unsigned error "
#                   "%.6f" % err)
#     else:
#         plt.title(title + " of tstat dataset")
#     plt.legend(loc="best", shadow=False, scatterpoints=1)
#     plt.axis([-4, 4, -1.5, 1.5])

# plt.show()
n_samples = x.shape[0]
# We center the data and compute the sample covariance matrix.
x -= np.mean(x, axis=0)
cov_matrix = np.dot(x.T, x) / n_samples
#for eigenvector in pca.components_:
    #print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
print("end")

