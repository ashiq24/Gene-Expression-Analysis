# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

from random import shuffle
filename='lung.tab'
with open(filename) as f:
    lines=f.read().splitlines()
    lines=lines[3:]
    shuffle(lines)
    shuffle(lines)
    shuffle(lines)
    shuffle(lines)
    shuffle(lines)
    shuffle(lines)
    lines=[ i.split('\t') for i in lines]
    categories=[i[0] for i in lines]
    lines=[i[1:] for i in lines]
    lines=[[float(j) for j in i] for i in lines]

#print lines[0][0:10]
#print categories[0:10]


for i in range(len(lines[0])):
    base=0
    for j in range(len(lines)):
        if(base>lines[j][i]):
            base=lines[j][i]
    base=base*-1
    for j in range(len(lines)):
        lines[j][i]+=base
for i in range(len(lines[0])):
    base=1
    for j in range(len(lines)):
        if(base<lines[j][i]):
            base=lines[j][i]
    for j in range(len(lines)):
        lines[j][i]/=base
#%%
Tenfold=0
equ1x=[]
equ1y=[]
chixa=[1000,1500,1700,2000,2200,2500,3000]
pcaya = [50,170,80,150,120,90,180]
solx=0
soly=0
for j in range(0,6,1):
#	chix=2254
#	pcay=115
	chix=chixa[j]
	pcay=pcaya[j]
	for i in range(0,10,1):
		print("testing ---->")
		lines=lines[180:203]+lines[0:180]
		categories=categories[180:203]+categories[0:180]
		print(str(pcay)+" "+str(chix))
		C=sorted(list(set(categories)))
		Map= dict((c, i) for i, c in enumerate(C))
		Y=[]
		for i in categories:
		    Y.append(Map[i])
		#print Y[0:10]

		from sklearn.feature_selection import SelectKBest
		from sklearn.feature_selection import chi2,f_classif, mutual_info_classif
		from sklearn import decomposition
		f=SelectKBest(chi2, k=chix)
		xlines = f.fit_transform(lines[0:180], Y[0:180])
		mast=f.get_support()

		feachers=[]
		i=0
		for bool in mast:
		    if bool:
		     feachers.append(i)
		    i+=1
		pca = decomposition.PCA(n_components=pcay)
		pca.fit(xlines)
		xlines = pca.transform(xlines)
		xlines=xlines.tolist()
		length=len(xlines[0])

		from keras.utils import np_utils
		Y=np_utils.to_categorical(Y)
		X=np.array(xlines)
		print(len(xlines[0]))

		from keras.models import Sequential
		from keras.layers import Dense
		from keras.layers import Dropout ,ActivityRegularization,LeakyReLU
		from keras.regularizers import l1_l2,l1

		model = Sequential()
		model.add(Dense(2*length, activation='linear',input_shape=(length,)))
		model.add(LeakyReLU(alpha=.1))
		model.add(ActivityRegularization(l2=0.001))
		model.add(Dense(length, activation='linear'))
		model.add(LeakyReLU(alpha=.1))
		model.add(ActivityRegularization(l2=0.001))
		model.add(Dense(5, activation='softmax'))
		model.compile(optimizer='adam',
			      loss='categorical_crossentropy',
			      metrics=['accuracy'])
		model.fit(X,Y[0:180],epochs=17,verbose=0)


		testx=[]
		for j in range(180,203,1):
		    m=[lines[j][i] for i in feachers]
		    testx.append(m)

		testx=pca.transform(testx)

		#testx=np.array(testx)

		score = model.evaluate(testx,Y[180:] ,verbose=0)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])
		Tenfold+=score[1]
	Tenfold=Tenfold/10.0
	print(str(chix)+" "+str(pcay)+" "+str(100*(1-Tenfold)))
	equ1y.append((1-Tenfold)*100)
	equ1x.append([chix*chix,pcay*pcay, chix*pcay,chix,pcay])
	Tenfold=0
print(100*Tenfold)
#%%
z,_,_,_=np.linalg.lstsq(equ1x,equ1y)
print(z)
z=z.tolist()
X=[ [ 2*z[0] , z[2] ], 
   [ z[2] , 2*z[1] ] ]
m=np.linalg.solve(X,[-z[3],-z[4]])
print("roots are --->"+str(m))
if 4*z[0]*z[1]-z[2]*z[2] <= 0:
	print("problem in curve saddle point")
elif z[0]<0 :
    print("not good surface")
else :
    print(m)
#%%

for j in range(0,1,1):
	chix=int(m[0])
	pcay=int(m[1])
	for i in range(0,10,1):
		print("testing Main---->")
		lines=lines[180:203]+lines[0:180]
		categories=categories[180:203]+categories[0:180]
		print(str(pcay)+" "+str(chix))
		C=sorted(list(set(categories)))
		Map= dict((c, i) for i, c in enumerate(C))
		Y=[]
		for i in categories:
		    Y.append(Map[i])
		from sklearn.feature_selection import SelectKBest
		from sklearn.feature_selection import chi2,f_classif, mutual_info_classif
		from sklearn import decomposition
		f=SelectKBest(chi2, k=chix)
		xlines = f.fit_transform(lines[0:180], Y[0:180])
		mast=f.get_support()

		feachers=[]
		i=0
		for bool in mast:
		    if bool:
		     feachers.append(i)
		    i+=1
		pca = decomposition.PCA(n_components=pcay)
		pca.fit(xlines)
		xlines = pca.transform(xlines)
		xlines=xlines.tolist()
		length=len(xlines[0])

		from keras.utils import np_utils
		Y=np_utils.to_categorical(Y)
		X=np.array(xlines)
		print(len(xlines[0]))

		from keras.models import Sequential
		from keras.layers import Dense
		from keras.layers import Dropout ,ActivityRegularization,LeakyReLU
		from keras.regularizers import l1_l2,l1

		model = Sequential()
		model.add(Dense(2*length, activation='linear',input_shape=(length,)))
		model.add(LeakyReLU(alpha=.1))
		model.add(ActivityRegularization(l2=0.001))
		model.add(Dense(length, activation='linear'))
		model.add(LeakyReLU(alpha=.1))
		model.add(ActivityRegularization(l2=0.001))
		model.add(Dense(5, activation='softmax'))
		model.compile(optimizer='adam',
			      loss='categorical_crossentropy',
			      metrics=['accuracy'])
		#model.summary()

		model.fit(X,Y[0:180],epochs=17,verbose=0)


		testx=[]
		for j in range(180,203,1):
		    m=[lines[j][i] for i in feachers]
		    testx.append(m)

		testx=pca.transform(testx)

		#testx=np.array(testx)

		score = model.evaluate(testx,Y[180:] ,verbose=0)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])
		Tenfold+=score[1]
	print(str(chix)+" main res "+ str(Tenfold))
	Tenfold=0

'''
Testing ------------>
Using TensorFlow backend.
180
2017-11-17 01:45:29.617080: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-11-17 01:45:29.617127: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-11-17 01:45:29.617139: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
('Test loss:', 0.41868191957473755)
('Test accuracy:', 0.95652174949645996)
Testing ------------>
180
('Test loss:', 0.41147738695144653)
('Test accuracy:', 0.95652174949645996)
Testing ------------>
180
('Test loss:', 0.47543856501579285)
('Test accuracy:', 0.91304349899291992)
Testing ------------>
180
('Test loss:', 0.34119802713394165)
('Test accuracy:', 1.0)
Testing ------------>
180
('Test loss:', 0.440123051404953)
('Test accuracy:', 1.0)
Testing ------------>
180
('Test loss:', 0.43540623784065247)
('Test accuracy:', 0.95652174949645996)
Testing ------------>
180
('Test loss:', 0.45053654909133911)
('Test accuracy:', 0.95652174949645996)
Testing ------------>
180
('Test loss:', 0.48132658004760742)
('Test accuracy:', 0.95652174949645996)
Testing ------------>
180
('Test loss:', 0.53477483987808228)
('Test accuracy:', 0.91304349899291992)
Testing ------------>
180
('Test loss:', 0.43573707342147827)
('Test accuracy:', 0.95652174949645996)
9.56521749496
'''
