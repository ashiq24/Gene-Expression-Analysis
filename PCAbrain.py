import numpy as np

from random import shuffle
from sklearn.utils import shuffle as s
filename='braintumor.tab'
with open(filename) as f:
    lines=f.read().splitlines()
    lines=lines[3:]
    shuffle(lines)
    #lines=s(lines,random_state=103)
    shuffle(lines)
    #lines=s(lines,random_state=109)
    shuffle(lines)
    shuffle(lines)
    #lines=s(lines,random_state=503)
    shuffle(lines)
    lines=[ i.split('\t') for i in lines]
    categories=[i[len(i)-1] for i in lines]
    lines=[i[0:len(i)-1] for i in lines]
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
Tenfold=0
for i in range(0,10,1):
	print("Testing ------------>")
	lines=lines[36:40]+lines[0:36]
	categories=categories[36:40]+categories[0:36]

	C=sorted(list(set(categories)))
	Map= dict((c, i) for i, c in enumerate(C))
	Y=[]
	for i in categories:
	    Y.append(Map[i])
	print (Y[0:20])

	from sklearn.feature_selection import SelectKBest
	from sklearn.feature_selection import chi2,f_classif, mutual_info_classif
	from sklearn import decomposition
	f=SelectKBest(chi2, k=1700)
	xlines = f.fit_transform(lines[0:36], Y[0:36])
	mast=f.get_support()

	feachers=[]
	i=0
	for bool in mast:
	    if bool: feachers.append(i)
	    i+=1
	pca = decomposition.PCA(n_components=200)
	pca.fit(xlines)
	xlines = pca.transform(xlines)
	xlines=xlines.tolist()
	length=len(xlines[0])
	from keras.utils import np_utils
	Y=np_utils.to_categorical(Y)
	X=np.array(xlines)
	print('nuber of gene '+str(len(xlines[0])))

	from keras.models import Sequential
	from keras.layers import Dense
	from keras.layers import Dropout ,ActivityRegularization,LeakyReLU
	from keras.regularizers import l1_l2,l1

	model = Sequential()
	model.add(Dense(7*length, activation='linear',input_shape=(length,)))
	model.add(LeakyReLU(alpha=.1))
	model.add(ActivityRegularization(l2=0.001))
	model.add(Dense(7*length, activation='linear'))
	model.add(LeakyReLU(alpha=.1))
	model.add(ActivityRegularization(l2=0.001))
	#model.add(ActivityRegularization(l2=0.001))
	#model.add(Dense(800, activation='elu'))
	#model.add(ActivityRegularization(l2=0.01))
	#model.add(Dense(500, activation='elu'))
	#model.add(Dense(100, activation='elu'))
	#model.add(Dense(100, activation='elu'))
	#model.add(Dense(100, activation='elu'))
	model.add(Dense(5, activation='softmax'))
	model.compile(optimizer='adam',
		      loss='categorical_crossentropy',
		      metrics=['accuracy'])
	#model.summary()

	model.fit(X,Y[0:36],epochs=15)


	testx=[]
	for j in range(36,40,1):
	    m=[lines[j][i] for i in feachers]
	    testx.append(m)
		
	testx=pca.transform(testx)
		
	#testx=np.array(testx)


	score = model.evaluate(testx,Y[36:] ,verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	Tenfold+=score[1]
print(Tenfold)
'''
Testing ------------>
[3, 4, 3, 3, 3, 2, 0, 4, 4, 3, 1, 1, 2, 2, 2, 4, 2, 0, 2, 3]
Using TensorFlow backend.
nuber of gene 36
Epoch 1/15
2017-11-17 02:22:29.425835: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-11-17 02:22:29.425870: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-11-17 02:22:29.425881: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
36/36 [==============================] - 0s - loss: 5.0089 - acc: 0.1667     
Epoch 2/15
36/36 [==============================] - 0s - loss: 4.4041 - acc: 0.6944     
Epoch 3/15
36/36 [==============================] - 0s - loss: 3.8386 - acc: 0.8056     
Epoch 4/15
36/36 [==============================] - 0s - loss: 3.1732 - acc: 0.9167     
Epoch 5/15
36/36 [==============================] - 0s - loss: 3.1673 - acc: 0.9722     
Epoch 6/15
36/36 [==============================] - 0s - loss: 3.0467 - acc: 1.0000     
Epoch 7/15
36/36 [==============================] - 0s - loss: 2.9328 - acc: 1.0000     
Epoch 8/15
36/36 [==============================] - 0s - loss: 2.7851 - acc: 1.0000     
Epoch 9/15
36/36 [==============================] - 0s - loss: 2.6789 - acc: 1.0000     
Epoch 10/15
36/36 [==============================] - 0s - loss: 2.5307 - acc: 1.0000     
Epoch 11/15
36/36 [==============================] - 0s - loss: 2.4713 - acc: 1.0000     
Epoch 12/15
36/36 [==============================] - 0s - loss: 2.5625 - acc: 1.0000     
Epoch 13/15
36/36 [==============================] - 0s - loss: 2.4229 - acc: 1.0000     
Epoch 14/15
36/36 [==============================] - 0s - loss: 2.3089 - acc: 1.0000     
Epoch 15/15
36/36 [==============================] - 0s - loss: 2.2148 - acc: 1.0000     
('Test loss:', 0.83480960130691528)
('Test accuracy:', 0.75)
Testing ------------>
[3, 3, 1, 4, 3, 4, 3, 3, 3, 2, 0, 4, 4, 3, 1, 1, 2, 2, 2, 4]
nuber of gene 36
Epoch 1/15
36/36 [==============================] - 0s - loss: 5.3462 - acc: 0.3611     
Epoch 2/15
36/36 [==============================] - 0s - loss: 4.1580 - acc: 0.7222     
Epoch 3/15
36/36 [==============================] - 0s - loss: 4.1512 - acc: 0.8889     
Epoch 4/15
36/36 [==============================] - 0s - loss: 3.5007 - acc: 0.8611     
Epoch 5/15
36/36 [==============================] - 0s - loss: 3.4907 - acc: 0.9167     
Epoch 6/15
36/36 [==============================] - 0s - loss: 3.1793 - acc: 0.9444     
Epoch 7/15
36/36 [==============================] - 0s - loss: 2.8884 - acc: 0.9444     
Epoch 8/15
36/36 [==============================] - 0s - loss: 2.8669 - acc: 0.9722     
Epoch 9/15
36/36 [==============================] - 0s - loss: 2.8940 - acc: 1.0000     
Epoch 10/15
36/36 [==============================] - 0s - loss: 2.7975 - acc: 1.0000     
Epoch 11/15
36/36 [==============================] - 0s - loss: 2.6024 - acc: 1.0000     
Epoch 12/15
36/36 [==============================] - 0s - loss: 2.4089 - acc: 1.0000     
Epoch 13/15
36/36 [==============================] - 0s - loss: 2.3694 - acc: 1.0000     
Epoch 14/15
36/36 [==============================] - 0s - loss: 2.3112 - acc: 1.0000     
Epoch 15/15
36/36 [==============================] - 0s - loss: 2.3688 - acc: 1.0000     
('Test loss:', 1.0858933925628662)
('Test accuracy:', 0.75)
Testing ------------>
[4, 2, 1, 0, 3, 3, 1, 4, 3, 4, 3, 3, 3, 2, 0, 4, 4, 3, 1, 1]
nuber of gene 36
Epoch 1/15
36/36 [==============================] - 0s - loss: 4.9711 - acc: 0.1111     
Epoch 2/15
36/36 [==============================] - 0s - loss: 4.0309 - acc: 0.5278     
Epoch 3/15
36/36 [==============================] - 0s - loss: 3.7187 - acc: 0.7778     
Epoch 4/15
36/36 [==============================] - 0s - loss: 3.3352 - acc: 0.8611     
Epoch 5/15
36/36 [==============================] - 0s - loss: 3.0541 - acc: 0.9167     
Epoch 6/15
36/36 [==============================] - 0s - loss: 3.0479 - acc: 0.9444     
Epoch 7/15
36/36 [==============================] - 0s - loss: 2.8865 - acc: 0.9444     
Epoch 8/15
36/36 [==============================] - 0s - loss: 2.7941 - acc: 0.9722     
Epoch 9/15
36/36 [==============================] - 0s - loss: 2.4621 - acc: 1.0000     
Epoch 10/15
36/36 [==============================] - 0s - loss: 2.5715 - acc: 1.0000     
Epoch 11/15
36/36 [==============================] - 0s - loss: 2.4797 - acc: 1.0000     
Epoch 12/15
36/36 [==============================] - 0s - loss: 2.3869 - acc: 1.0000     
Epoch 13/15
36/36 [==============================] - 0s - loss: 2.1892 - acc: 1.0000     
Epoch 14/15
36/36 [==============================] - 0s - loss: 2.1674 - acc: 1.0000     
Epoch 15/15
36/36 [==============================] - 0s - loss: 2.1923 - acc: 1.0000     
('Test loss:', 0.53693175315856934)
('Test accuracy:', 1.0)
Testing ------------>
[4, 3, 4, 0, 4, 2, 1, 0, 3, 3, 1, 4, 3, 4, 3, 3, 3, 2, 0, 4]
nuber of gene 36
Epoch 1/15
36/36 [==============================] - 0s - loss: 5.0805 - acc: 0.0833     
Epoch 2/15
36/36 [==============================] - 0s - loss: 4.2149 - acc: 0.7222     
Epoch 3/15
36/36 [==============================] - 0s - loss: 3.7573 - acc: 0.7778     
Epoch 4/15
36/36 [==============================] - 0s - loss: 3.5112 - acc: 0.8333     
Epoch 5/15
36/36 [==============================] - 0s - loss: 3.3148 - acc: 0.8611     
Epoch 6/15
36/36 [==============================] - 0s - loss: 3.0291 - acc: 0.9167     
Epoch 7/15
36/36 [==============================] - 0s - loss: 2.8316 - acc: 0.9722     
Epoch 8/15
36/36 [==============================] - 0s - loss: 2.7013 - acc: 1.0000     
Epoch 9/15
36/36 [==============================] - 0s - loss: 2.5751 - acc: 1.0000     
Epoch 10/15
36/36 [==============================] - 0s - loss: 2.5574 - acc: 1.0000     
Epoch 11/15
36/36 [==============================] - 0s - loss: 2.4563 - acc: 1.0000     
Epoch 12/15
36/36 [==============================] - 0s - loss: 2.4113 - acc: 1.0000     
Epoch 13/15
36/36 [==============================] - 0s - loss: 2.3084 - acc: 1.0000     
Epoch 14/15
36/36 [==============================] - 0s - loss: 2.2467 - acc: 1.0000     
Epoch 15/15
36/36 [==============================] - 0s - loss: 2.1906 - acc: 1.0000     
('Test loss:', 0.77721208333969116)
('Test accuracy:', 0.75)
Testing ------------>
[2, 2, 2, 1, 4, 3, 4, 0, 4, 2, 1, 0, 3, 3, 1, 4, 3, 4, 3, 3]
nuber of gene 36
Epoch 1/15
36/36 [==============================] - 0s - loss: 5.0147 - acc: 0.1667     
Epoch 2/15
36/36 [==============================] - 0s - loss: 4.2367 - acc: 0.6111     
Epoch 3/15
36/36 [==============================] - 0s - loss: 3.8337 - acc: 0.8056     
Epoch 4/15
36/36 [==============================] - 0s - loss: 3.4442 - acc: 0.8333     
Epoch 5/15
36/36 [==============================] - 0s - loss: 3.3071 - acc: 0.8333     
Epoch 6/15
36/36 [==============================] - 0s - loss: 3.0739 - acc: 0.9167     
Epoch 7/15
36/36 [==============================] - 0s - loss: 2.9945 - acc: 0.9722     
Epoch 8/15
36/36 [==============================] - 0s - loss: 2.8122 - acc: 1.0000     
Epoch 9/15
36/36 [==============================] - 0s - loss: 2.5462 - acc: 1.0000     
Epoch 10/15
36/36 [==============================] - 0s - loss: 2.6101 - acc: 1.0000     
Epoch 11/15
36/36 [==============================] - 0s - loss: 2.3820 - acc: 1.0000     
Epoch 12/15
36/36 [==============================] - 0s - loss: 2.2517 - acc: 1.0000     
Epoch 13/15
36/36 [==============================] - 0s - loss: 2.2615 - acc: 1.0000     
Epoch 14/15
36/36 [==============================] - 0s - loss: 2.1776 - acc: 1.0000     
Epoch 15/15
36/36 [==============================] - 0s - loss: 2.1534 - acc: 1.0000     
('Test loss:', 0.90824496746063232)
('Test accuracy:', 0.75)
Testing ------------>
[1, 4, 3, 4, 2, 2, 2, 1, 4, 3, 4, 0, 4, 2, 1, 0, 3, 3, 1, 4]
nuber of gene 36
Epoch 1/15
36/36 [==============================] - 0s - loss: 4.9525 - acc: 0.2500     
Epoch 2/15
36/36 [==============================] - 0s - loss: 4.1251 - acc: 0.7500     
Epoch 3/15
36/36 [==============================] - 0s - loss: 3.5743 - acc: 0.8611     
Epoch 4/15
36/36 [==============================] - 0s - loss: 3.3349 - acc: 0.9167     
Epoch 5/15
36/36 [==============================] - 0s - loss: 3.1181 - acc: 0.9167     
Epoch 6/15
36/36 [==============================] - 0s - loss: 2.8794 - acc: 0.9444     
Epoch 7/15
36/36 [==============================] - 0s - loss: 2.7078 - acc: 0.9444     
Epoch 8/15
36/36 [==============================] - 0s - loss: 2.6466 - acc: 0.9444     
Epoch 9/15
36/36 [==============================] - 0s - loss: 2.5810 - acc: 0.9722     
Epoch 10/15
36/36 [==============================] - 0s - loss: 2.5128 - acc: 0.9722     
Epoch 11/15
36/36 [==============================] - 0s - loss: 2.3856 - acc: 0.9722     
Epoch 12/15
36/36 [==============================] - 0s - loss: 2.3109 - acc: 1.0000     
Epoch 13/15
36/36 [==============================] - 0s - loss: 2.2097 - acc: 1.0000     
Epoch 14/15
36/36 [==============================] - 0s - loss: 2.1265 - acc: 1.0000     
Epoch 15/15
36/36 [==============================] - 0s - loss: 2.0316 - acc: 1.0000     
('Test loss:', 1.0280159711837769)
('Test accuracy:', 0.75)
Testing ------------>
[2, 0, 2, 3, 1, 4, 3, 4, 2, 2, 2, 1, 4, 3, 4, 0, 4, 2, 1, 0]
nuber of gene 36
Epoch 1/15
36/36 [==============================] - 0s - loss: 5.1282 - acc: 0.2222     
Epoch 2/15
36/36 [==============================] - 0s - loss: 4.2968 - acc: 0.5556     
Epoch 3/15
36/36 [==============================] - 0s - loss: 3.6647 - acc: 0.8889     
Epoch 4/15
36/36 [==============================] - 0s - loss: 3.5306 - acc: 0.8611     
Epoch 5/15
36/36 [==============================] - 0s - loss: 3.2548 - acc: 0.9167     
Epoch 6/15
36/36 [==============================] - 0s - loss: 3.1802 - acc: 0.9167     
Epoch 7/15
36/36 [==============================] - 0s - loss: 2.9915 - acc: 0.9167     
Epoch 8/15
36/36 [==============================] - 0s - loss: 2.8776 - acc: 0.9167     
Epoch 9/15
36/36 [==============================] - 0s - loss: 2.6954 - acc: 0.9167     
Epoch 10/15
36/36 [==============================] - 0s - loss: 2.7191 - acc: 0.9167     
Epoch 11/15
36/36 [==============================] - 0s - loss: 2.5944 - acc: 0.9167     
Epoch 12/15
36/36 [==============================] - 0s - loss: 2.4360 - acc: 0.9722     
Epoch 13/15
36/36 [==============================] - 0s - loss: 2.3970 - acc: 0.9722     
Epoch 14/15
36/36 [==============================] - 0s - loss: 2.3603 - acc: 0.9722     
Epoch 15/15
36/36 [==============================] - 0s - loss: 2.2120 - acc: 1.0000     
('Test loss:', 0.34228140115737915)
('Test accuracy:', 1.0)
Testing ------------>
[2, 2, 2, 4, 2, 0, 2, 3, 1, 4, 3, 4, 2, 2, 2, 1, 4, 3, 4, 0]
nuber of gene 36
Epoch 1/15
36/36 [==============================] - 0s - loss: 5.7558 - acc: 0.0833     
Epoch 2/15
36/36 [==============================] - 0s - loss: 4.6073 - acc: 0.5000     
Epoch 3/15
36/36 [==============================] - 0s - loss: 4.0735 - acc: 0.6667     
Epoch 4/15
36/36 [==============================] - 0s - loss: 3.6959 - acc: 0.9167     
Epoch 5/15
36/36 [==============================] - 0s - loss: 3.5405 - acc: 0.9444     
Epoch 6/15
36/36 [==============================] - 0s - loss: 3.2588 - acc: 0.9167     
Epoch 7/15
36/36 [==============================] - 0s - loss: 3.1714 - acc: 0.9722     
Epoch 8/15
36/36 [==============================] - 0s - loss: 2.9238 - acc: 1.0000     
Epoch 9/15
36/36 [==============================] - 0s - loss: 2.9773 - acc: 1.0000     
Epoch 10/15
36/36 [==============================] - 0s - loss: 2.7762 - acc: 1.0000     
Epoch 11/15
36/36 [==============================] - 0s - loss: 2.6489 - acc: 1.0000     
Epoch 12/15
36/36 [==============================] - 0s - loss: 2.5754 - acc: 1.0000     
Epoch 13/15
36/36 [==============================] - 0s - loss: 2.3245 - acc: 1.0000     
Epoch 14/15
36/36 [==============================] - 0s - loss: 2.3140 - acc: 1.0000     
Epoch 15/15
36/36 [==============================] - 0s - loss: 2.2721 - acc: 1.0000     
('Test loss:', 1.2376250028610229)
('Test accuracy:', 0.5)
Testing ------------>
[4, 3, 1, 1, 2, 2, 2, 4, 2, 0, 2, 3, 1, 4, 3, 4, 2, 2, 2, 1]
nuber of gene 36
Epoch 1/15
36/36 [==============================] - 0s - loss: 4.7402 - acc: 0.2222     
Epoch 2/15
36/36 [==============================] - 0s - loss: 3.9507 - acc: 0.8056     
Epoch 3/15
36/36 [==============================] - 0s - loss: 3.7260 - acc: 0.9167     
Epoch 4/15
36/36 [==============================] - 0s - loss: 3.3814 - acc: 0.9444     
Epoch 5/15
36/36 [==============================] - 0s - loss: 3.0964 - acc: 0.9722     
Epoch 6/15
36/36 [==============================] - 0s - loss: 2.9157 - acc: 0.9722     
Epoch 7/15
36/36 [==============================] - 0s - loss: 2.8720 - acc: 1.0000     
Epoch 8/15
36/36 [==============================] - 0s - loss: 2.5870 - acc: 0.9722     
Epoch 9/15
36/36 [==============================] - 0s - loss: 2.5358 - acc: 0.9722     
Epoch 10/15
36/36 [==============================] - 0s - loss: 2.4813 - acc: 1.0000     
Epoch 11/15
36/36 [==============================] - 0s - loss: 2.3169 - acc: 1.0000     
Epoch 12/15
36/36 [==============================] - 0s - loss: 2.2620 - acc: 1.0000     
Epoch 13/15
36/36 [==============================] - 0s - loss: 2.1448 - acc: 1.0000     
Epoch 14/15
36/36 [==============================] - 0s - loss: 1.9707 - acc: 1.0000     
Epoch 15/15
36/36 [==============================] - 0s - loss: 2.0952 - acc: 1.0000     
('Test loss:', 0.99939537048339844)
('Test accuracy:', 0.75)
Testing ------------>
[3, 2, 0, 4, 4, 3, 1, 1, 2, 2, 2, 4, 2, 0, 2, 3, 1, 4, 3, 4]
nuber of gene 36
Epoch 1/15
36/36 [==============================] - 0s - loss: 5.1105 - acc: 0.2222     
Epoch 2/15
36/36 [==============================] - 0s - loss: 4.3191 - acc: 0.6667     
Epoch 3/15
36/36 [==============================] - 0s - loss: 3.8883 - acc: 0.7778     
Epoch 4/15
36/36 [==============================] - 0s - loss: 3.4791 - acc: 0.7778     
Epoch 5/15
36/36 [==============================] - 0s - loss: 3.1876 - acc: 0.8611     
Epoch 6/15
36/36 [==============================] - 0s - loss: 3.1059 - acc: 0.8611     
Epoch 7/15
36/36 [==============================] - 0s - loss: 2.7956 - acc: 0.8889     
Epoch 8/15
36/36 [==============================] - 0s - loss: 2.7012 - acc: 0.9167     
Epoch 9/15
36/36 [==============================] - 0s - loss: 2.7149 - acc: 0.9722     
Epoch 10/15
36/36 [==============================] - 0s - loss: 2.6133 - acc: 1.0000     
Epoch 11/15
36/36 [==============================] - 0s - loss: 2.4330 - acc: 1.0000     
Epoch 12/15
36/36 [==============================] - 0s - loss: 2.3797 - acc: 1.0000     
Epoch 13/15
36/36 [==============================] - 0s - loss: 2.1231 - acc: 1.0000     
Epoch 14/15
36/36 [==============================] - 0s - loss: 2.3028 - acc: 1.0000     
Epoch 15/15
36/36 [==============================] - 0s - loss: 2.1970 - acc: 1.0000     
('Test loss:', 0.55999457836151123)
('Test accuracy:', 1.0)
8.0

'''
