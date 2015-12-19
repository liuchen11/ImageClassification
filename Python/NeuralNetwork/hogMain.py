import sys
sys.path.append('../')
import theano
import numpy as np
import scipy.io as sio

from cnnModel import *
from mlpModel import *
from loadImg import *

if __name__=='__main__':
	if len(sys.argv)!=2:
		print 'Usage: python hogMain.py [-cnn|-mlp]'
		exit(0)
	[cnnFeature,hogFeature,labels]=getDataY()
	[cnnTest,hogTest]=getTestData()
	assert(cnnFeature.shape[0]==labels.shape[0])
	assert(hogFeature.shape[0]==labels.shape[0])
	##Normalize##
	hogFeature=hogFeature-np.ones([hogFeature.shape[0],1])*hogFeature.mean(axis=0)
	hogFeature=hogFeature/(np.ones([hogFeature.shape[0],1])*hogFeature.std(axis=0))

	modelType=sys.argv[1]

	instanceNum=labels.shape[0]
	trainInstanceNum=int(instanceNum*0.8)
	testInstanceNum=int(hogTest.shape[0])
	randIndex=np.random.permutation(range(instanceNum))
	trainSetX=hogFeature[randIndex[:trainInstanceNum]]
	trainSetY=labels[randIndex[:trainInstanceNum]]
	valSetX=hogFeature[randIndex[trainInstanceNum:]]
	valSetY=labels[randIndex[trainInstanceNum:]]
	testX=hogTest

	trainSet={}
	valSet={}
	testSet={}
	trainSet['y']=np.asarray(trainSetY,
		dtype=theano.config.floatX).reshape(trainInstanceNum)-1
	valSet['y']=np.asarray(valSetY,
		dtype=theano.config.floatX).reshape(instanceNum-trainInstanceNum)-1

	if modelType=='-cnn':
		trainSet['x']=np.asarray(trainSetX,
			dtype=theano.config.floatX).reshape(trainInstanceNum,32,13,13)
		valSet['x']=np.asarray(valSetX,
			dtype=theano.config.floatX).reshape(instanceNum-trainInstanceNum,32,13,13)
		testSet['x']=np.asarray(testX,
			dtype=theano.config.floatX).reshape(testInstanceNum,32,13,13)
		model=CNNModel(
			shape=(25,32,13,13),
			filters=((4,4),(3,3),(0,0)),
			poolings=((2,2),(1,1),(1,1)),
			features=(64,64,128),
			wdecay=0.0,
			categories=4,
			dropoutRate=(0.0,0.0,0.0),
			learningRate=0.03,
			name='naiveCNN'
			)
		[maxValAcc4C,test4CResult,test2CResult]=model.training(trainSet,valSet,testSet,25)
		sio.savemat('HogCNN4C.mat',{'4CResult':list(np.transpose(test4CResult))})
		sio.savemat('HogCNN2C.mat',{'2CResult':list(np.transpose(test2CResult))})
	else:
		trainSet['x']=np.asarray(trainSetX,
			dtype=theano.config.floatX).reshape(trainInstanceNum,5408)
		valSet['x']=np.asarray(valSetX,
			dtype=theano.config.floatX).reshape(instanceNum-trainInstanceNum,5408)
		testSet['x']=np.asarray(testX,
			dtype=theano.config.floatX).reshape(testInstanceNum,5408)
		model=MLPModel(
			shape=(25,5408),
			neurons=(8,),
			wdecay=0.0,
			categories=4,
			dropoutRate=(0.0,0.0),
			learningRate=0.03,
			name='naiveMLP'
			)
		[maxValAcc4C,test4CResult,test2CResult]=model.training(trainSet,valSet,testSet,5)
		sio.savemat('HogMLP4C.mat',{'4CResult':list(np.transpose(test4CResult))})
		sio.savemat('HogMLP2C.mat',{'2CResult':list(np.transpose(test2CResult))})
