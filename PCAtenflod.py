# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
import numpy as np
from random import shuffle
filename='prostata.tab'
with open(filename) as f:
    lines=f.read().splitlines()
    lines=lines[3:]
    shuffle(lines)
    shuffle(lines)
    shuffle(lines)
    shuffle(lines)
    shuffle(lines)
    lines=[ i.split('\t') for i in lines]
    categories=[i[0] for i in lines]
    lines=[i[1:] for i in lines]
    lines=[[float(j) for j in i] for i in lines]
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
chixa=[500,1000,1500,2000,2500,3000,3500]
solx=0
soly=0
for j in range(0,7,1):
#	chix=2254
#	pcay=115
	chix=chixa[j]
	pcay=200
	for i in range(0,10,1):
		print("testing ---->")
		lines=lines[90:102]+lines[0:90]
		categories=categories[90:102]+categories[0:90]
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
		xlines = f.fit_transform(lines[0:90], Y[0:90])
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
		model.add(Dense(2, activation='softmax'))
		model.compile(optimizer='adam',
			      loss='categorical_crossentropy',
			      metrics=['accuracy'])
		model.fit(X,Y[0:90],epochs=17,verbose=0)


		testx=[]
		for j in range(90,102,1):
		    m=[lines[j][i] for i in feachers]
		    testx.append(m)

		testx=pca.transform(testx)

		#testx=np.array(testx)

		score = model.evaluate(testx,Y[90:] ,verbose=0)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])
		Tenfold+=score[1]
	Tenfold=Tenfold/10.0
	print(str(chix)+" "+str(pcay)+" "+str(100*(1-Tenfold)))
	equ1y.append((1-Tenfold)*100)
	equ1x.append(chix/1000)
	Tenfold=0
print(100*Tenfold)
#%%
z=np.polyfit(equ1x,equ1y,2)
print(z)
m=-1*z[1]/(2*z[0])
print("roots are --->"+str(m))
if z[0]<=0 :
	print("-------> problem with curve ")
	m=4.500
	
#%%
for j in range(0,1,1):
	chix=int(m*1000)
	pcay=200
	if m>3.500:
		m=3.500
	if chix<0:
		continue
	for i in range(0,10,1):
		print("testing Main---->")
		lines=lines[90:102]+lines[0:90]
		categories=categories[90:102]+categories[0:90]
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
		xlines = f.fit_transform(lines[0:90], Y[0:90])
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
		model.add(Dense(2, activation='softmax'))
		model.compile(optimizer='adam',
			      loss='categorical_crossentropy',
			      metrics=['accuracy'])
		#model.summary()

		model.fit(X,Y[0:90],epochs=17,verbose=0)


		testx=[]
		for j in range(90,102,1):
		    m=[lines[j][i] for i in feachers]
		    testx.append(m)

		testx=pca.transform(testx)

		#testx=np.array(testx)

		score = model.evaluate(testx,Y[90:] ,verbose=0)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])
		Tenfold+=score[1]
	print(str(chix)+" main res "+ str(Tenfold))
	Tenfold=0
'''
Testing ------------>
Using TensorFlow backend.
90
Epoch 1/15
2017-11-17 01:34:27.300714: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-11-17 01:34:27.300754: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-11-17 01:34:27.300773: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
90/90 [==============================] - 0s - loss: 2.5737 - acc: 0.6556
Epoch 2/15
90/90 [==============================] - 0s - loss: 2.2238 - acc: 0.8333
Epoch 3/15
90/90 [==============================] - 0s - loss: 1.9968 - acc: 0.8778
Epoch 4/15
90/90 [==============================] - 0s - loss: 1.8036 - acc: 0.9333
Epoch 5/15
90/90 [==============================] - 0s - loss: 1.6558 - acc: 0.9333
Epoch 6/15
90/90 [==============================] - 0s - loss: 1.5371 - acc: 0.9444
Epoch 7/15
90/90 [==============================] - 0s - loss: 1.4234 - acc: 0.9444
Epoch 8/15
90/90 [==============================] - 0s - loss: 1.3650 - acc: 0.9556
Epoch 9/15
90/90 [==============================] - 0s - loss: 1.2709 - acc: 0.9556
Epoch 10/15
90/90 [==============================] - 0s - loss: 1.1835 - acc: 0.9778
Epoch 11/15
90/90 [==============================] - 0s - loss: 1.0998 - acc: 0.9778
Epoch 12/15
90/90 [==============================] - 0s - loss: 1.0615 - acc: 0.9889
Epoch 13/15
90/90 [==============================] - 0s - loss: 0.9762 - acc: 0.9889
Epoch 14/15
90/90 [==============================] - 0s - loss: 0.9465 - acc: 0.9889
Epoch 15/15
90/90 [==============================] - 0s - loss: 0.9064 - acc: 1.0000
('Test loss:', 0.3137681782245636)
('Test accuracy:', 1.0)
Testing ------------>
90
Epoch 1/15
90/90 [==============================] - 0s - loss: 2.4621 - acc: 0.4000
Epoch 2/15
90/90 [==============================] - 0s - loss: 2.1296 - acc: 0.7667
Epoch 3/15
90/90 [==============================] - 0s - loss: 1.9147 - acc: 0.8333
Epoch 4/15
90/90 [==============================] - 0s - loss: 1.7339 - acc: 0.9111
Epoch 5/15
90/90 [==============================] - 0s - loss: 1.5904 - acc: 0.9333
Epoch 6/15
90/90 [==============================] - 0s - loss: 1.4962 - acc: 0.9667
Epoch 7/15
90/90 [==============================] - 0s - loss: 1.3513 - acc: 0.9667
Epoch 8/15
90/90 [==============================] - 0s - loss: 1.2994 - acc: 0.9667
Epoch 9/15
90/90 [==============================] - 0s - loss: 1.2290 - acc: 0.9667
Epoch 10/15
90/90 [==============================] - 0s - loss: 1.1648 - acc: 0.9778
Epoch 11/15
90/90 [==============================] - 0s - loss: 1.0964 - acc: 0.9778
Epoch 12/15
90/90 [==============================] - 0s - loss: 1.0429 - acc: 0.9778
Epoch 13/15
90/90 [==============================] - 0s - loss: 0.9629 - acc: 0.9889
Epoch 14/15
90/90 [==============================] - 0s - loss: 0.9349 - acc: 1.0000
Epoch 15/15
90/90 [==============================] - 0s - loss: 0.8613 - acc: 1.0000
('Test loss:', 0.48312452435493469)
('Test accuracy:', 1.0)
Testing ------------>
90
Epoch 1/15
90/90 [==============================] - 0s - loss: 2.7055 - acc: 0.4222
Epoch 2/15
90/90 [==============================] - 0s - loss: 2.3185 - acc: 0.7667
Epoch 3/15
90/90 [==============================] - 0s - loss: 2.0676 - acc: 0.8111
Epoch 4/15
90/90 [==============================] - 0s - loss: 1.8142 - acc: 0.8889
Epoch 5/15
90/90 [==============================] - 0s - loss: 1.6723 - acc: 0.9111
Epoch 6/15
90/90 [==============================] - 0s - loss: 1.5397 - acc: 0.9667
Epoch 7/15
90/90 [==============================] - 0s - loss: 1.4531 - acc: 0.9889
Epoch 8/15
90/90 [==============================] - 0s - loss: 1.3396 - acc: 0.9889
Epoch 9/15
90/90 [==============================] - 0s - loss: 1.3164 - acc: 1.0000
Epoch 10/15
90/90 [==============================] - 0s - loss: 1.2189 - acc: 1.0000
Epoch 11/15
90/90 [==============================] - 0s - loss: 1.1432 - acc: 1.0000
Epoch 12/15
90/90 [==============================] - 0s - loss: 1.0748 - acc: 1.0000
Epoch 13/15
90/90 [==============================] - 0s - loss: 1.0412 - acc: 1.0000
Epoch 14/15
90/90 [==============================] - 0s - loss: 0.9746 - acc: 1.0000
Epoch 15/15
90/90 [==============================] - 0s - loss: 0.9135 - acc: 1.0000
('Test loss:', 0.50662946701049805)
('Test accuracy:', 0.91666668653488159)
Testing ------------>
90
Epoch 1/15
90/90 [==============================] - 0s - loss: 2.4662 - acc: 0.5444
Epoch 2/15
90/90 [==============================] - 0s - loss: 2.0903 - acc: 0.8000
Epoch 3/15
90/90 [==============================] - 0s - loss: 1.9088 - acc: 0.8444
Epoch 4/15
90/90 [==============================] - 0s - loss: 1.7227 - acc: 0.8889
Epoch 5/15
90/90 [==============================] - 0s - loss: 1.5632 - acc: 0.9000
Epoch 6/15
90/90 [==============================] - 0s - loss: 1.4506 - acc: 0.9667
Epoch 7/15
90/90 [==============================] - 0s - loss: 1.3624 - acc: 0.9889
Epoch 8/15
90/90 [==============================] - 0s - loss: 1.3032 - acc: 0.9889
Epoch 9/15
90/90 [==============================] - 0s - loss: 1.1902 - acc: 0.9889
Epoch 10/15
90/90 [==============================] - 0s - loss: 1.1567 - acc: 0.9889
Epoch 11/15
90/90 [==============================] - 0s - loss: 1.0959 - acc: 0.9889
Epoch 12/15
90/90 [==============================] - 0s - loss: 1.0262 - acc: 0.9889
Epoch 13/15
90/90 [==============================] - 0s - loss: 0.9780 - acc: 0.9889
Epoch 14/15
90/90 [==============================] - 0s - loss: 0.9238 - acc: 0.9889
Epoch 15/15
90/90 [==============================] - 0s - loss: 0.8718 - acc: 0.9889
('Test loss:', 0.5593528151512146)
('Test accuracy:', 0.91666668653488159)
Testing ------------>
90
Epoch 1/15
90/90 [==============================] - 0s - loss: 2.4145 - acc: 0.5333
Epoch 2/15
90/90 [==============================] - 0s - loss: 2.0754 - acc: 0.8000
Epoch 3/15
90/90 [==============================] - 0s - loss: 1.8585 - acc: 0.8444
Epoch 4/15
90/90 [==============================] - 0s - loss: 1.7040 - acc: 0.9222
Epoch 5/15
90/90 [==============================] - 0s - loss: 1.5486 - acc: 0.9556
Epoch 6/15
90/90 [==============================] - 0s - loss: 1.4147 - acc: 0.9556
Epoch 7/15
90/90 [==============================] - 0s - loss: 1.3581 - acc: 0.9556
Epoch 8/15
90/90 [==============================] - 0s - loss: 1.2509 - acc: 0.9667
Epoch 9/15
90/90 [==============================] - 0s - loss: 1.1679 - acc: 0.9778
Epoch 10/15
90/90 [==============================] - 0s - loss: 1.0982 - acc: 0.9889
Epoch 11/15
90/90 [==============================] - 0s - loss: 1.0336 - acc: 0.9889
Epoch 12/15
90/90 [==============================] - 0s - loss: 0.9823 - acc: 0.9889
Epoch 13/15
90/90 [==============================] - 0s - loss: 0.9302 - acc: 0.9889
Epoch 14/15
90/90 [==============================] - 0s - loss: 0.8892 - acc: 1.0000
Epoch 15/15
90/90 [==============================] - 0s - loss: 0.8562 - acc: 1.0000
('Test loss:', 0.39252769947052002)
('Test accuracy:', 1.0)
Testing ------------>
90
Epoch 1/15
90/90 [==============================] - 0s - loss: 2.3920 - acc: 0.5556
Epoch 2/15
90/90 [==============================] - 0s - loss: 2.0282 - acc: 0.7667
Epoch 3/15
90/90 [==============================] - 0s - loss: 1.7828 - acc: 0.8000
Epoch 4/15
90/90 [==============================] - 0s - loss: 1.6854 - acc: 0.8889
Epoch 5/15
90/90 [==============================] - 0s - loss: 1.5634 - acc: 0.9333
Epoch 6/15
90/90 [==============================] - 0s - loss: 1.4620 - acc: 0.9444
Epoch 7/15
90/90 [==============================] - 0s - loss: 1.3268 - acc: 0.9667
Epoch 8/15
90/90 [==============================] - 0s - loss: 1.2563 - acc: 0.9778
Epoch 9/15
90/90 [==============================] - 0s - loss: 1.1713 - acc: 0.9778
Epoch 10/15
90/90 [==============================] - 0s - loss: 1.0898 - acc: 0.9778
Epoch 11/15
90/90 [==============================] - 0s - loss: 1.0312 - acc: 0.9778
Epoch 12/15
90/90 [==============================] - 0s - loss: 1.0070 - acc: 0.9778
Epoch 13/15
90/90 [==============================] - 0s - loss: 0.9313 - acc: 0.9778
Epoch 14/15
90/90 [==============================] - 0s - loss: 0.8967 - acc: 1.0000
Epoch 15/15
90/90 [==============================] - 0s - loss: 0.8517 - acc: 1.0000
('Test loss:', 0.68456220626831055)
('Test accuracy:', 0.83333331346511841)
Testing ------------>
90
Epoch 1/15
90/90 [==============================] - 0s - loss: 2.2758 - acc: 0.5667
Epoch 2/15
90/90 [==============================] - 0s - loss: 1.9700 - acc: 0.7111
Epoch 3/15
90/90 [==============================] - 0s - loss: 1.7347 - acc: 0.8333
Epoch 4/15
90/90 [==============================] - 0s - loss: 1.5823 - acc: 0.9000
Epoch 5/15
90/90 [==============================] - 0s - loss: 1.4630 - acc: 0.9333
Epoch 6/15
90/90 [==============================] - 0s - loss: 1.3713 - acc: 0.9444
Epoch 7/15
90/90 [==============================] - 0s - loss: 1.2609 - acc: 0.9778
Epoch 8/15
90/90 [==============================] - 0s - loss: 1.1529 - acc: 0.9778
Epoch 9/15
90/90 [==============================] - 0s - loss: 1.1106 - acc: 0.9778
Epoch 10/15
90/90 [==============================] - 0s - loss: 1.0376 - acc: 0.9889
Epoch 11/15
90/90 [==============================] - 0s - loss: 0.9757 - acc: 0.9889
Epoch 12/15
90/90 [==============================] - 0s - loss: 0.9313 - acc: 1.0000
Epoch 13/15
90/90 [==============================] - 0s - loss: 0.8861 - acc: 1.0000
Epoch 14/15
90/90 [==============================] - 0s - loss: 0.8390 - acc: 1.0000
Epoch 15/15
90/90 [==============================] - 0s - loss: 0.7986 - acc: 1.0000
('Test loss:', 0.4600728452205658)
('Test accuracy:', 0.91666668653488159)
Testing ------------>
90
Epoch 1/15
90/90 [==============================] - 0s - loss: 2.5327 - acc: 0.6333
Epoch 2/15
90/90 [==============================] - 0s - loss: 2.1795 - acc: 0.7667
Epoch 3/15
90/90 [==============================] - 0s - loss: 1.9143 - acc: 0.9111
Epoch 4/15
90/90 [==============================] - 0s - loss: 1.7985 - acc: 0.9333
Epoch 5/15
90/90 [==============================] - 0s - loss: 1.6389 - acc: 0.9556
Epoch 6/15
90/90 [==============================] - 0s - loss: 1.5076 - acc: 0.9556
Epoch 7/15
90/90 [==============================] - 0s - loss: 1.3827 - acc: 0.9667
Epoch 8/15
90/90 [==============================] - 0s - loss: 1.3550 - acc: 0.9778
Epoch 9/15
90/90 [==============================] - 0s - loss: 1.2764 - acc: 0.9778
Epoch 10/15
90/90 [==============================] - 0s - loss: 1.2089 - acc: 0.9778
Epoch 11/15
90/90 [==============================] - 0s - loss: 1.1036 - acc: 1.0000
Epoch 12/15
90/90 [==============================] - 0s - loss: 1.0492 - acc: 1.0000
Epoch 13/15
90/90 [==============================] - 0s - loss: 0.9788 - acc: 1.0000
Epoch 14/15
90/90 [==============================] - 0s - loss: 0.9447 - acc: 1.0000
Epoch 15/15
90/90 [==============================] - 0s - loss: 0.9062 - acc: 1.0000
('Test loss:', 0.46651023626327515)
('Test accuracy:', 0.91666668653488159)
Testing ------------>
90
Epoch 1/15
90/90 [==============================] - 0s - loss: 2.5913 - acc: 0.5444
Epoch 2/15
90/90 [==============================] - 0s - loss: 2.1896 - acc: 0.7333
Epoch 3/15
90/90 [==============================] - 0s - loss: 2.0046 - acc: 0.8333
Epoch 4/15
90/90 [==============================] - 0s - loss: 1.8131 - acc: 0.9111
Epoch 5/15
90/90 [==============================] - 0s - loss: 1.6987 - acc: 0.9333
Epoch 6/15
90/90 [==============================] - 0s - loss: 1.5716 - acc: 0.9778
Epoch 7/15
90/90 [==============================] - 0s - loss: 1.4659 - acc: 0.9778
Epoch 8/15
90/90 [==============================] - 0s - loss: 1.3924 - acc: 0.9778
Epoch 9/15
90/90 [==============================] - 0s - loss: 1.2838 - acc: 0.9889
Epoch 10/15
90/90 [==============================] - 0s - loss: 1.2149 - acc: 0.9889
Epoch 11/15
90/90 [==============================] - 0s - loss: 1.1453 - acc: 0.9889
Epoch 12/15
90/90 [==============================] - 0s - loss: 1.0879 - acc: 1.0000
Epoch 13/15
90/90 [==============================] - 0s - loss: 1.0291 - acc: 1.0000
Epoch 14/15
90/90 [==============================] - 0s - loss: 0.9588 - acc: 1.0000
Epoch 15/15
90/90 [==============================] - 0s - loss: 0.8996 - acc: 1.0000
('Test loss:', 0.39304384589195251)
('Test accuracy:', 0.91666668653488159)
Testing ------------>
90
Epoch 1/15
90/90 [==============================] - 0s - loss: 2.2269 - acc: 0.6778
Epoch 2/15
90/90 [==============================] - 0s - loss: 1.9222 - acc: 0.8778
Epoch 3/15
90/90 [==============================] - 0s - loss: 1.7142 - acc: 0.9222
Epoch 4/15
90/90 [==============================] - 0s - loss: 1.5723 - acc: 0.9444
Epoch 5/15
90/90 [==============================] - 0s - loss: 1.4414 - acc: 0.9444
Epoch 6/15
90/90 [==============================] - 0s - loss: 1.3023 - acc: 0.9444
Epoch 7/15
90/90 [==============================] - 0s - loss: 1.2292 - acc: 0.9556
Epoch 8/15
90/90 [==============================] - 0s - loss: 1.1465 - acc: 0.9667
Epoch 9/15
90/90 [==============================] - 0s - loss: 1.0896 - acc: 0.9778
Epoch 10/15
90/90 [==============================] - 0s - loss: 1.0235 - acc: 0.9889
Epoch 11/15
90/90 [==============================] - 0s - loss: 0.9691 - acc: 0.9889
Epoch 12/15
90/90 [==============================] - 0s - loss: 0.9245 - acc: 1.0000
Epoch 13/15
90/90 [==============================] - 0s - loss: 0.8619 - acc: 1.0000
Epoch 14/15
90/90 [==============================] - 0s - loss: 0.8176 - acc: 1.0000
Epoch 15/15
90/90 [==============================] - 0s - loss: 0.7761 - acc: 1.0000
('Test loss:', 0.40977036952972412)
('Test accuracy:', 1.0)
9.41666674614
'''
