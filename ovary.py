import numpy as np

from scipy.io.arff import loadarff
from random import shuffle
lines,meta = loadarff('Ovarian.arff')
lines=lines.tolist()
#print lines[0]
shuffle(lines)
shuffle(lines)
shuffle(lines)
shuffle(lines)
shuffle(lines)
shuffle(lines)
#lines=[ i.split('\t') for i in lines]
categories=[i[len(i)-1] for i in lines]
lines=[i[0:len(i)-1] for i in lines]
lines=[[float(j) for j in i] for i in lines]

#from random import shuffle
#filename='braintumor.tab'
#with open(filename) as f:
#    lines=f.read().splitlines()
#    print len(lines)
#    lines=lines[3:]
#    shuffle(lines)
#    shuffle(lines)
#    shuffle(lines)
#    shuffle(lines)
#    shuffle(lines)
#    shuffle(lines)
#    lines=[ i.split('\t') for i in lines]
#    categories=[i[len(i)-1] for i in lines]
#    lines=[i[0:len(i)-1] for i in lines]
#    lines=[[float(j) for j in i] for i in lines]

print len(lines[0])
print len(lines)
print categories[0:10]


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
	lines=lines[228:253]+lines[0:228]
	categories=categories[228:253]+categories[0:228]

	C=sorted(list(set(categories)))
	Map= dict((c, i) for i, c in enumerate(C))
	Y=[]
	for i in categories:
	    Y.append(Map[i])
	print Y[0:20]

	from sklearn.feature_selection import SelectKBest
	from sklearn.feature_selection import chi2,f_classif, mutual_info_classif
	f=SelectKBest(chi2, k=1500)
	xlines = f.fit_transform(lines[0:228], Y[0:228])
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
	model.add(Dense(2000, activation='linear',W_regularizer = l1(0.0001),input_shape=(1500,)))
	model.add(LeakyReLU(alpha=.001))
	#model.add(ActivityRegularization(l2=0.001))
	model.add(Dense(1000, activation='linear',W_regularizer = l1(0.001)))
	model.add(LeakyReLU(alpha=.001))
	#model.add(ActivityRegularization(l2=0.001))
	#model.add(Dense(800, activation='elu'))
	#model.add(ActivityRegularization(l2=0.01))
	#model.add(Dense(500, activation='elu'))
	#model.add(Dense(100, activation='elu'))
	#model.add(Dense(100, activation='elu'))
	#model.add(Dense(100, activation='elu'))
	model.add(Dense(2, activation='softmax'))
	model.compile(optimizer='adam',
		      loss='categorical_crossentropy',
		      metrics=['accuracy'])
	#model.summary()

	model.fit(X,Y[0:228],epochs=17, verbose=0)


	testx=[]
	for j in range(228,253,1):
	    m=[lines[j][i] for i in feachers]
	    testx.append(m)
		
	testx=np.array(testx)

	score = model.evaluate(testx,Y[228:] ,verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	Tenfold+=score[1]
print Tenfold
'''
15154
253
['Normal', 'Cancer', 'Normal', 'Cancer', 'Normal', 'Cancer', 'Cancer', 'Cancer', 'Cancer', 'Cancer']
Testing ------------>
[1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1]
Using TensorFlow backend.
1500
2017-11-14 01:48:45.834511: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-11-14 01:48:45.834556: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-11-14 01:48:45.834567: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
('Test loss:', 0.08147081732749939)
('Test accuracy:', 1.0)
Testing ------------>
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0]
1500
('Test loss:', 0.076367132365703583)
('Test accuracy:', 1.0)
Testing ------------>
[1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
1500
('Test loss:', 0.12766732275485992)
('Test accuracy:', 1.0)
Testing ------------>
[1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1]
1500
('Test loss:', 0.073547564446926117)
('Test accuracy:', 1.0)
Testing ------------>
[1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
1500
('Test loss:', 0.1070878803730011)
('Test accuracy:', 1.0)
Testing ------------>
[0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
1500
('Test loss:', 0.1073354035615921)
('Test accuracy:', 1.0)
Testing ------------>
[1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
1500
('Test loss:', 0.09076286107301712)
('Test accuracy:', 1.0)
Testing ------------>
[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1]
1500
('Test loss:', 0.10702294111251831)
('Test accuracy:', 1.0)
Testing ------------>
[1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
1500
('Test loss:', 0.086188025772571564)
('Test accuracy:', 1.0)
Testing ------------>
[0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0]
1500
('Test loss:', 0.13325780630111694)
('Test accuracy:', 0.95999997854232788)
9.95999997854

l1 regularizers
('Test loss:', 2.6864919662475586)
('Test accuracy:', 0.95999997854232788)
Testing ------------>
[1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]
1500
('Test loss:', 2.6015028953552246)
('Test accuracy:', 1.0)
Testing ------------>
[0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
1500
('Test loss:', 2.5110735893249512)
('Test accuracy:', 0.95999997854232788)
Testing ------------>
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0]
1500
('Test loss:', 2.152857780456543)
('Test accuracy:', 1.0)
Testing ------------>
[1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]
1500
('Test loss:', 1.9885928630828857)
('Test accuracy:', 1.0)
Testing ------------>
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
1500
('Test loss:', 2.6888594627380371)
('Test accuracy:', 0.95999997854232788)
Testing ------------>
[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0]
1500
('Test loss:', 2.7091226577758789)
('Test accuracy:', 1.0)
Testing ------------>
[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0]
1500
('Test loss:', 2.2165019512176514)
('Test accuracy:', 1.0)
Testing ------------>
[1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0]
1500
('Test loss:', 2.9430100917816162)
('Test accuracy:', 0.95999997854232788)
Testing ------------>
[1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1]
1500
('Test loss:', 2.2741563320159912)
('Test accuracy:', 1.0)
9.83999991417

9.80000001192

'''
