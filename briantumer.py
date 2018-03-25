import numpy as np

from random import shuffle
from sklearn.utils import shuffle as s
filename='braintumor.tab'
with open(filename) as f:
    lines=f.read().splitlines()
    lines=lines[3:]
    shuffle(lines)
    lines=s(lines,random_state=13)
    shuffle(lines)
    lines=s(lines,random_state=13)
    #shuffle(lines)
    #shuffle(lines)
    #shuffle(lines)
    #shuffle(lines)
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
	print Y[0:20]

	from sklearn.feature_selection import SelectKBest
	from sklearn.feature_selection import chi2,f_classif, mutual_info_classif
	f=SelectKBest(chi2, k=1500)
	xlines = f.fit_transform(lines[0:36], Y[0:36])
	mast=f.get_support()

	feachers=[]
	i=0
	for bool in mast:
	    if bool:
		feachers.append(i)
	    i+=1

	from keras.utils import np_utils
	Y=np_utils.to_categorical(Y)
	X=np.array(xlines)
	print(len(xlines[0]))

	from keras.models import Sequential
	from keras.layers import Dense
	from keras.layers import Dropout ,ActivityRegularization,LeakyReLU
	from keras.regularizers import l1_l2,l1

	model = Sequential()
	model.add(Dense(2000, activation='linear',input_shape=(1500,)))
	model.add(LeakyReLU(alpha=.001))
	#model.add(ActivityRegularization(l2=0.001))
	model.add(Dense(1000, activation='linear'))
	model.add(LeakyReLU(alpha=.001))
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

	model.fit(X,Y[0:36],epochs=20,verbose=0)


	testx=[]
	for j in range(36,40,1):
	    m=[lines[j][i] for i in feachers]
	    testx.append(m)
		
	testx=np.array(testx)

	score = model.evaluate(testx,Y[36:] ,verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	Tenfold+=score[1]
print Tenfold
'''
Testing ------------>
[4, 2, 4, 3, 4, 4, 4, 3, 2, 2, 0, 3, 2, 3, 2, 0, 3, 3, 1, 2]
Using TensorFlow backend.
1500
2017-11-13 23:52:04.098186: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-11-13 23:52:04.098229: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-11-13 23:52:04.098241: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
('Test loss:', 0.67286217212677002)
('Test accuracy:', 0.75)
Testing ------------>
[1, 3, 0, 1, 4, 2, 4, 3, 4, 4, 4, 3, 2, 2, 0, 3, 2, 3, 2, 0]
1500
('Test loss:', 0.60612279176712036)
('Test accuracy:', 1.0)
Testing ------------>
[1, 3, 3, 3, 1, 3, 0, 1, 4, 2, 4, 3, 4, 4, 4, 3, 2, 2, 0, 3]
1500
('Test loss:', 0.17993755638599396)
('Test accuracy:', 1.0)
Testing ------------>
[4, 4, 2, 4, 1, 3, 3, 3, 1, 3, 0, 1, 4, 2, 4, 3, 4, 4, 4, 3]
1500
('Test loss:', 0.59443247318267822)
('Test accuracy:', 1.0)
Testing ------------>
[0, 1, 1, 2, 4, 4, 2, 4, 1, 3, 3, 3, 1, 3, 0, 1, 4, 2, 4, 3]
1500
('Test loss:', 0.48373505473136902)
('Test accuracy:', 1.0)
Testing ------------>
[4, 2, 4, 2, 0, 1, 1, 2, 4, 4, 2, 4, 1, 3, 3, 3, 1, 3, 0, 1]
1500
('Test loss:', 0.86922669410705566)
('Test accuracy:', 0.75)
Testing ------------>
[3, 3, 1, 2, 4, 2, 4, 2, 0, 1, 1, 2, 4, 4, 2, 4, 1, 3, 3, 3]
1500
('Test loss:', 0.38708955049514771)
('Test accuracy:', 1.0)
Testing ------------>
[2, 3, 2, 0, 3, 3, 1, 2, 4, 2, 4, 2, 0, 1, 1, 2, 4, 4, 2, 4]
1500
('Test loss:', 0.63730823993682861)
('Test accuracy:', 1.0)
Testing ------------>
[2, 2, 0, 3, 2, 3, 2, 0, 3, 3, 1, 2, 4, 2, 4, 2, 0, 1, 1, 2]
1500
('Test loss:', 0.27130389213562012)
('Test accuracy:', 1.0)
Testing ------------>
[4, 4, 4, 3, 2, 2, 0, 3, 2, 3, 2, 0, 3, 3, 1, 2, 4, 2, 4, 2]
1500
('Test loss:', 1.1065397262573242)
('Test accuracy:', 0.5)
9.0

l1 regularizers
('Test loss:', 14.170034408569336)
('Test accuracy:', 0.75)
Testing ------------>
[2, 1, 3, 2, 1, 4, 2, 3, 2, 2, 3, 4, 1, 4, 4, 4, 3, 1, 4, 2]
1500
('Test loss:', 14.40976619720459)
('Test accuracy:', 0.75)
Testing ------------>
[3, 4, 1, 2, 2, 1, 3, 2, 1, 4, 2, 3, 2, 2, 3, 4, 1, 4, 4, 4]
1500
('Test loss:', 13.309581756591797)
('Test accuracy:', 1.0)
Testing ------------>
[3, 4, 3, 4, 3, 4, 1, 2, 2, 1, 3, 2, 1, 4, 2, 3, 2, 2, 3, 4]
1500
('Test loss:', 14.132896423339844)
('Test accuracy:', 0.75)
Testing ------------>
[3, 2, 3, 0, 3, 4, 3, 4, 3, 4, 1, 2, 2, 1, 3, 2, 1, 4, 2, 3]
1500
('Test loss:', 13.693099975585938)
('Test accuracy:', 0.75)
Testing ------------>
[0, 0, 2, 2, 3, 2, 3, 0, 3, 4, 3, 4, 3, 4, 1, 2, 2, 1, 3, 2]
1500
('Test loss:', 14.317960739135742)
('Test accuracy:', 0.75)
Testing ------------>
[1, 4, 3, 0, 0, 0, 2, 2, 3, 2, 3, 0, 3, 4, 3, 4, 3, 4, 1, 2]
1500
('Test loss:', 15.630725860595703)
('Test accuracy:', 0.75)
Testing ------------>
[3, 1, 4, 2, 1, 4, 3, 0, 0, 0, 2, 2, 3, 2, 3, 0, 3, 4, 3, 4]
1500
('Test loss:', 13.510410308837891)
('Test accuracy:', 1.0)
Testing ------------>
[1, 4, 4, 4, 3, 1, 4, 2, 1, 4, 3, 0, 0, 0, 2, 2, 3, 2, 3, 0]
1500
('Test loss:', 13.346370697021484)
('Test accuracy:', 1.0)
Testing ------------>
[2, 2, 3, 4, 1, 4, 4, 4, 3, 1, 4, 2, 1, 4, 3, 0, 0, 0, 2, 2]
1500
('Test loss:', 14.057449340820312)
('Test accuracy:', 0.75)
8.25
7.75

'''
