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
	sio.savemat('4CResultOverfeat.mat',{'4CResult':list(np.transpose(test4CResult))})
	sio.savemat('2CResultOverfeat.mat',{'2CResult':list(np.transpose(test2CResult))})
