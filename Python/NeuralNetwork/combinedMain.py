import sys
sys.path.append('../')
import theano
import numpy as np
import scipy.io as sio

from combinedModel import *
from loadImg import *

if __name__=='__main__':
	[cnnFeature,hogFeature,labels]=getDataY()
	[cnnTest,hogTest]=getTestData()
	assert(cnnFeature.shape[0]==labels.shape[0])
	assert(hogFeature.shape[0]==labels.shape[0])

	mode='binary' if len(sys.argv)==2 and argv[1]=='-binary' else 'multiClass'

	instanceNum=labels.shape[0]
	trainInstanceNum=int(instanceNum*0.8)
	testInstanceNum=cnnTest.shape[0]
	randIndex=np.random.permutation(range(instanceNum))
	trainMlpX=cnnFeature[randIndex[:trainInstanceNum]]
	trainCnnX=hogFeature[randIndex[:trainInstanceNum]]
	trainY=labels[randIndex[:trainInstanceNum]]
	valMlpX=cnnFeature[randIndex[trainInstanceNum:]]
	valCnnX=hogFeature[randIndex[trainInstanceNum:]]
	valY=labels[randIndex[trainInstanceNum:]]
	testMlpX=cnnTest
	testCnnX=hogTest

	trainSet={}
	valSet={}
	testSet={}
	trainSet['mlpX']=np.asarray(trainMlpX[:,:-1],
		dtype=theano.config.floatX).reshape(trainInstanceNum,36864)
	trainSet['cnnX']=np.asarray(trainCnnX,
		dtype=theano.config.floatX).reshape(trainInstanceNum,32,13,13)
	trainSet['y']=np.asarray(trainY,
		dtype=theano.config.floatX).reshape(trainInstanceNum)-1
	valSet['mlpX']=np.asarray(valMlpX[:,:-1],
		dtype=theano.config.floatX).reshape(instanceNum-trainInstanceNum,36864)
	valSet['cnnX']=np.asarray(valCnnX,
		dtype=theano.config.floatX).reshape(instanceNum-trainInstanceNum,32,13,13)
	valSet['y']=np.asarray(valY,
		dtype=theano.config.floatX).reshape(instanceNum-trainInstanceNum)-1
	testSet['mlpX']=np.asarray(testMlpX[:,:-1],
		dtype=theano.config.floatX).reshape(testInstanceNum,36864)
	testSet['cnnX']=np.asarray(testCnnX,
		dtype=theano.config.floatX).reshape(testInstanceNum,32,13,13)

	if mode=='multiClass':
		model=CombinedModel(
			mlpShape=(25,36864),
			mlpNeurons=(512,),
			mlpDropout=(0.0,0.0,),
			cnnShape=(25,32,13,13),
			cnnFeatures=(64,64,128),
			cnnFilters=((4,4),(3,3),(0,0)),
			cnnPoolings=((2,2),(1,1),(0,0)),
			cnnDropout=(0.0,0.0,0.0),
			wdecay=0.0,
			categories=4,
			learningRate=0.03,
			name='combined'
			)
		[maxValAcc4C,test4CResult,test2CResult]=model.training(trainSet,valSet,testSet,5)
		saveResult('Comb4C.mat',{'Predict4C':list(np.transpose(test4CResult))})
		saveResult('Comb2C.mat',{'Predict2C':list(np.transpose(test2CResult))})
	elif mode=='binary':
		trainSet['y']=(trainSet['y']+1)/4
		valSet['y']=(valSet['y']+1)/4
		model=CombinedModel(
			mlpShape=(25,36864),
			mlpNeurons=(512,),
			mlpDropout=(0.0,0.0,),
			cnnShape=(25,32,13,13),
			cnnFeatures=(64,64,128),
			cnnFilters=((4,4),(3,3),(0,0)),
			cnnPoolings=((2,2),(1,1),(0,0)),
			cnnDropout=(0.0,0.0,0.0),
			wdecay=0.0,
			categories=2,
			learningRate=0.03,
			name='combined'			
			)
		[maxValAcc2C,test2CResult,_]=model.training(trainSet,valSet,testSet,5)
		saveResult('CombBinary.mat',{'PredictBinary':list(np.transpose(test2CResult))})
