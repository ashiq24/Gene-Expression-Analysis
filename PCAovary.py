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
	from sklearn import decomposition
	f=SelectKBest(chi2, k=1500)
	xlines = f.fit_transform(lines[0:228], Y[0:228])
	mast=f.get_support()

	feachers=[]
	i=0
	for bool in mast:
	    if bool:
		feachers.append(i)
	    i+=1
	pca = decomposition.PCA(n_components=400)
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
	model.summary()

	model.fit(X,Y[0:228],epochs=17)


	testx=[]
	for j in range(228,253,1):
	    m=[lines[j][i] for i in feachers]
	    testx.append(m)
	testx=pca.transform(testx)	
	#testx=np.array(testx)

	score = model.evaluate(testx,Y[228:] ,verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	Tenfold+=score[1]
print Tenfold
'''
('Test loss:', 0.10801710188388824)
('Test accuracy:', 1.0)
Testing ------------>
[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0]
228
('Test loss:', 0.11338154971599579)
('Test accuracy:', 1.0)
Testing ------------>
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0]
228
('Test loss:', 0.10748718678951263)
('Test accuracy:', 1.0)
Testing ------------>
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
228
('Test loss:', 0.11297857761383057)
('Test accuracy:', 1.0)
Testing ------------>
[1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]
228
('Test loss:', 0.11675851792097092)
('Test accuracy:', 1.0)
Testing ------------>
[0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1]
228
('Test loss:', 0.13849654793739319)
('Test accuracy:', 1.0)
Testing ------------>
[1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
228
('Test loss:', 0.12391471862792969)
('Test accuracy:', 1.0)
Testing ------------>
[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
228
('Test loss:', 0.11959076672792435)
('Test accuracy:', 1.0)
Testing ------------>
[1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
228
('Test loss:', 0.12871171534061432)
('Test accuracy:', 1.0)
Testing ------------>
[0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
228
('Test loss:', 0.10559659451246262)
('Test accuracy:', 1.0)
10.0
'''
