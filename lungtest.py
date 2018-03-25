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
Tenfold=0
for i in range(0,10,1):
	print("Testing ------------>")
	lines=lines[180:203]+lines[0:180]
	categories=categories[180:203]+categories[0:180]

	C=sorted(list(set(categories)))
	Map= dict((c, i) for i, c in enumerate(C))
	Y=[]
	for i in categories:
	    Y.append(Map[i])
	#print Y[0:10]

	from sklearn.feature_selection import SelectKBest
	from sklearn.feature_selection import chi2,f_classif, mutual_info_classif
	f=SelectKBest(chi2, k=1500)
	xlines = f.fit_transform(lines[0:180], Y[0:180])
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

	model.fit(X,Y[0:180],epochs=17,verbose=0)


	testx=[]
	for j in range(180,203,1):
	    m=[lines[j][i] for i in feachers]
	    testx.append(m)
		
	testx=np.array(testx)

	score = model.evaluate(testx,Y[180:] ,verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	Tenfold+=score[1]
print Tenfold
'''
('Test loss:', 0.43751451373100281)
('Test accuracy:', 0.91304349899291992)
Testing ------------>
1500
('Test loss:', 0.24828876554965973)
('Test accuracy:', 0.95652174949645996)
Testing ------------>
1500
('Test loss:', 0.24842967092990875)
('Test accuracy:', 0.95652174949645996)
Testing ------------>
1500
('Test loss:', 0.28858271241188049)
('Test accuracy:', 0.95652174949645996)
Testing ------------>
1500
('Test loss:', 0.15425577759742737)
('Test accuracy:', 1.0)
Testing ------------>
1500
('Test loss:', 0.37546443939208984)
('Test accuracy:', 0.91304349899291992)
Testing ------------>
1500
('Test loss:', 0.29900991916656494)
('Test accuracy:', 0.95652174949645996)
Testing ------------>
1500
('Test loss:', 0.42058178782463074)
('Test accuracy:', 0.91304349899291992)
Testing ------------>
1500
('Test loss:', 0.42139846086502075)
('Test accuracy:', 0.86956518888473511)
Testing ------------>
1500
('Test loss:', 0.30647927522659302)
('Test accuracy:', 0.91304349899291992)
9.34782618284


l1 regularizers
('Test loss:', 2.2588150501251221)
('Test accuracy:', 1.0)
Testing ------------>
1500
('Test loss:', 2.3005430698394775)
('Test accuracy:', 0.91304349899291992)
Testing ------------>
1500
('Test loss:', 2.0778422355651855)
('Test accuracy:', 0.91304349899291992)
Testing ------------>
1500
('Test loss:', 1.8524458408355713)
('Test accuracy:', 1.0)
Testing ------------>
1500
('Test loss:', 2.2371315956115723)
('Test accuracy:', 0.91304349899291992)
Testing ------------>
1500
('Test loss:', 2.7008991241455078)
('Test accuracy:', 0.86956518888473511)
Testing ------------>
1500
('Test loss:', 2.4871151447296143)
('Test accuracy:', 1.0)
Testing ------------>
1500
('Test loss:', 2.4609012603759766)
('Test accuracy:', 0.86956518888473511)
Testing ------------>
1500
('Test loss:', 2.2585697174072266)
('Test accuracy:', 0.95652174949645996)
Testing ------------>
1500
('Test loss:', 2.1583173274993896)
('Test accuracy:', 1.0)
9.43478262424
9.34782612324

'''
