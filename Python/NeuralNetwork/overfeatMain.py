import sys
sys.path.append('../')
import theano
import numpy as np
import scipy.io as sio

from cnnModel import *
from mlpModel import *
from loadImg import *

if __name__=='__main__':
	[cnnFeature,hogFeature,labels]=getDataY()
	[cnnTest,hogTest]=getTestData()
	assert(cnnFeature.shape[0]==labels.shape[0])
	assert(hogFeature.shape[0]==labels.shape[0])

	mode='binary' if len(sys.argv)==2 and sys.argv[1]=='-binary' else 'multiClass'

	instanceNum=labels.shape[0]
	trainInstanceNum=int(instanceNum*0.8)
	testInstanceNum=cnnTest.shape[0]
	randIndex=np.random.permutation(range(instanceNum))
	trainSetX=cnnFeature[randIndex[:trainInstanceNum]]
	trainSetY=labels[randIndex[:trainInstanceNum]]
	validateSetX=cnnFeature[randIndex[trainInstanceNum:]]
	validateSetY=labels[randIndex[trainInstanceNum:]]
	testX=cnnTest

	trainSet={}
	validateSet={}
	testSet={}
	trainSet['x']=np.asarray(trainSetX[:,:-1],
		dtype=theano.config.floatX).reshape(trainInstanceNum,36864)
	trainSet['y']=np.asarray(trainSetY,
		dtype=theano.config.floatX).reshape(trainInstanceNum)-1
	validateSet['x']=np.asarray(validateSetX[:,:-1],
		dtype=theano.config.floatX).reshape(instanceNum-trainInstanceNum,36864)
	validateSet['y']=np.asarray(validateSetY,
		dtype=theano.config.floatX).reshape(instanceNum-trainInstanceNum)-1
	testSet['x']=np.asarray(testX[:,:-1],
		dtype=theano.config.floatX).reshape(testInstanceNum,36864)

	if mode=='multiClass':
		model=MLPModel(
			shape=(25,36864),
			neurons=(512,),
			wdecay=0.0,
			categories=4,
			dropoutRate=(0.0,0.0),
			learningRate=0.02,
			name='naiveMLP'
			)
		[maxValAcc4C,test4CResult,test2CResult]=model.training(trainSet,validateSet,testSet,5)
		saveResult('OverfeatMLP4C.mat',{'Predict4C':list(np.transpose(test4CResult))})
		saveResult('OverfeatMLP2C.mat',{'Predict2C':list(np.transpose(test2CResult))})
	elif mode=='binary':
		trainSet['y']=(trainSet['y']+1)/4
		validateSet['y']=(validateSet['y']+1)/4

		model=MLPModel(
			shape=(25,36864),
			neurons=(512,),
			wdecay=0.0,
			categories=2,
			dropoutRate=(0.0,0.0),
			learningRate=0.02,
			name='binaryMLP'
			)
		[maxValAcc2C,test2CResult,_]=model.training(trainSet,validateSet,testSet,5)
		saveResult('OverfeatMLPBinary.mat',{'PredictBinary':list(np.transpose(test2CResult))})



