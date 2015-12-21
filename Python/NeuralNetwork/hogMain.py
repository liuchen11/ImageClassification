import sys
sys.path.append('../')
import theano
import numpy as np

from cnnModel import *
from mlpModel import *
from loadImg import *

if __name__=='__main__':
	if len(sys.argv)!=2 and len(sys.argv)!=3:
		print 'Usage: python hogMain.py [-cnn|-mlp] (-binary)'
		exit(0)
	[cnnFeature,hogFeature,labels]=getDataY()
	[cnnTest,hogTest]=getTestData()
	assert(cnnFeature.shape[0]==labels.shape[0])
	assert(hogFeature.shape[0]==labels.shape[0])

	modelType=sys.argv[1]
	mode='binary' if len(sys.argv)==3 and sys.argv[2]=='-binary' else 'multiClass'

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
		if mode=='multiClass':
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
			saveResult('HogCNN4C.mat',{'Predict4C':list(np.transpose(test4CResult))})
			saveResult('HogCNN2C.mat',{'Predict2C':list(np.transpose(test2CResult))})
		elif mode=='binary':
			trainSet['y']=(trainSet['y']+1)/4
			valSet['y']=(valSet['y']+1)/4
			model=cnnModel(
				shape=(25,32,13,13),
				filters=((4,4),(3,3),(0,0)),
				poolings=((2,2),(1,1),(1,1)),
				features=(64,64,128),
				wdecay=0.0,
				categories=2,
				dropoutRate=(0.0,0.0,0.0),
				learningRate=0.03,
				name='naiveCNN'
				)
			[maxValAcc2C,test2CResult,_]=model.training(trainSet,valSet,testSet,25)
			saveResult('HogCNNBinary.mat',{'PredictBinary':list(np.transpose(test2CResult))})
	else:
		trainSet['x']=np.asarray(trainSetX,
			dtype=theano.config.floatX).reshape(trainInstanceNum,5408)
		valSet['x']=np.asarray(valSetX,
			dtype=theano.config.floatX).reshape(instanceNum-trainInstanceNum,5408)
		testSet['x']=np.asarray(testX,
			dtype=theano.config.floatX).reshape(testInstanceNum,5408)
		if mode=='multiClass':
			model=MLPModel(
				shape=(25,5408),
				neurons=(512,),
				wdecay=0.0,
				categories=4,
				dropoutRate=(0.0,0.0),
				learningRate=0.03,
				name='naiveMLP'
				)
			[maxValAcc4C,test4CResult,test2CResult]=model.training(trainSet,valSet,testSet,25)
			saveResult('HogMLP4C.mat',{'Predict4C':list(np.transpose(test4CResult))})
			saveResult('HogMLP2C.mat',{'Predict2C':list(np.transpose(test2CResult))})
		elif mode=='binary':
			trainSet['y']=(trainSet['y']+1)/4
			valSet['y']=(valSet['y']+1)/4
			model=MLPModel(
				shape=(25,5408),
				neurons=(8,),
				wdecay=0.0,
				categories=2,
				dropoutRate=(0.0,0.0),
				learningRate=0.03,
				name='naiveMLP'
				)
			[maxValAcc2C,test2CResult,_]=model.training(trainSet,valSet,testSet,25)
			saveResult('HogMLPBinary.mat',{'PredictBinary':list(np.transpose(test2CResult))})
