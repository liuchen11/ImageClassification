import sys
import cPickle
import time
import theano

import numpy as np
import theano.tensor as T

from collections import defaultdict, OrderedDict
from convLayer import *
from hiddenLayer import *
from loadImg import *
from logisticRegression import *

sys.setrecursionlimit(40000)

def ReLU(x):
	return T.switch(x<0,0,x)

def as_floatX(variable):
	if isinstance(variable,float) or isinstance(variable,np.ndarray):
		return np.cast[theano.config.floatX](variable)
	return T.cast(variable,theano.config.floatX)

def AdadeltaUpdate(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9):
	'''
	>>>Apply ada-delta updates

	>>>type params: tuple or list
	>>>para params: parameters
	>>>type cost:
	>>>para cost:
	>>>type rho: float
	>>>para rho:
	>>>type epsilon: float
	>>>para epsilon:
	>>>type norm_lim: int
	>>>para norm_lim:
	'''
	updates=OrderedDict({})
	exp_sqr_grads=OrderedDict({})
	exp_sqr_update=OrderedDict({})
	g_params=[]
	for param in params:
		empty=np.zeros_like(param.get_value())
		exp_sqr_grads[param]=theano.shared(value=as_floatX(empty),name='exp_grad_%s'%param.name)
		exp_sqr_update[param]=theano.shared(value=as_floatX(empty),name='exp_grad_%s'%param.name)
		gp=T.grad(cost,param)
		g_params.append(gp)
	for param,gp in zip(params,g_params):
		exp_sg=exp_sqr_grads[param]
		exp_su=exp_sqr_update[param]
		update_exp_sg=rho*exp_sg+(1-rho)*T.sqr(gp)
		updates[exp_sg]=update_exp_sg
		
		step=-(T.sqrt(exp_su+epsilon)/T.sqrt(update_exp_sg+epsilon))*gp		
		stepped_param=param+step

		update_exp_su=rho*exp_su+(1-rho)*T.sqr(step)
		updates[exp_su]=update_exp_su

		col_norms=T.sqrt(T.sum(T.sqr(stepped_param),axis=0))
		desired_norms=T.clip(col_norms,0,T.sqrt(norm_lim))
		scale=desired_norms/(1e-7+col_norms)
		updates[param]=stepped_param*scale
	return updates

def sgdMomentum(params,cost,learningRate,momentum=0.9):
	'''
	>>>SGD optimizer with momentum
	>>>type params: tuple or list
	>>>para params: parameters of the model
	>>>type cost: T.tensorType
	>>>para cost: goal to be optimized
	>>>type learningRate: float
	>>>para learningRate: learning rate
	>>>type momentum: float
	>>>para momentum: momentum weight
	'''
	grads=T.grad(cost,params)
	updates=OrderedDict({})

	for param_i,grad_i in zip(params,grads):
		mparam_i=theano.shared(np.zeros(param_i.get_value().shape,dtype=theano.config.floatX),broadcastable=param_i.broadcastable)
		delta=momentum*mparam_i-learningRate*grad_i
		updates[mparam_i]=delta
		updates[param_i]=param_i+delta
	return updates

class MLPModel(object):

	def __init__(self,shape,neurons,wdecay,categories,dropoutRate,learningRate,name):
		'''
		>>>initalize the model

		>>>type shape: tuple or list of length 2
		>>>para shape: [batchSize, inputNeurons]
		>>>type neurons: tuple or list of int
		>>>para neurons: the number of neurons in each layer
		>>>type wdecay: float
		>>>para wdecay: weight decay
		>>>type categories: int
		>>>para categories: target categories
		>>>type dropoutRate: tuple of list of float
		>>>para dropoutRate: dropout rate of each layer
		>>>type learningRate: float
		>>>para learningRate: learning rate
		>>>type name: str
		>>>para name: model's name
		'''
		self.name=name
		self.categories=categories
		self.learningRate=learningRate
		rng=np.random.RandomState(254860)

		self.batchSize,self.inputSize=shape
		fcLayerNum=min(len(neurons),len(dropoutRate))
		print 'This is a mlp with %i layers'%(fcLayerNum)

		self.x=T.matrix('x')
		self.y=T.ivector('y')
		self.fcLayers=[]

		currentInput=self.x
		currentInNum=self.inputSize
		currentOutNum=neurons[0]

		for i in xrange(fcLayerNum):
			currentLayer=DropoutHiddenLayer(
				rng=rng,
				input=currentInput,
				n_in=currentInNum,
				n_out=currentOutNum,
				activation=ReLU,
				dropoutRate=dropoutRate[i]
				)
			self.fcLayers.append(currentLayer)
			currentInput=currentLayer.output
			if i!=fcLayerNum-1:
				currentInNum=neurons[i]
				currentOutNum=neurons[i+1]

		self.classifier=LogisticRegression(
			input=currentInput,
			n_in=currentOutNum,
			n_out=self.categories
			)

		weights=0
		for param in self.classifier.param:
			weights+=T.sum(T.sqr(param))

		self.params=self.classifier.param
		for subNets in self.fcLayers:
			self.params+=subNets.param

		self.cost=self.classifier.negative_log_likelihood(self.y)+wdecay*weights
		self.errors=self.classifier.errors(self.y)
		self.predict=self.classifier.predict
		grads=T.grad(self.cost,self.params)

		self.update=[
			(paramI,paramI-gradI*self.learningRate)
			for (paramI,gradI) in zip(self.params,grads)
		]
		self.adadeltaUpdate=AdadeltaUpdate(self.params,self.cost)
		self.momentumUpdate=sgdMomentum(self.params,self.cost,self.learningRate)

		print 'the model '+self.name+' constructed!'

	def training(self,trainSet,valSet,testSet,nEpoch):
		'''
		>>>training the model

		>>>trainSet/valSet: training/validation set
		>>>nEpoch: maximum iterations
		'''
		trainSize=trainSet['x'].shape[0]
		valSize=valSet['x'].shape[0]
		trainBatches=trainSize/self.batchSize
		valBatches=valSize/self.batchSize

		trainX=theano.shared(trainSet['x'],borrow=True)
		trainY=theano.shared(trainSet['y'],borrow=True)
		trainY=T.cast(trainY,'int32')
		valX=theano.shared(valSet['x'],borrow=True)
		valY=theano.shared(valSet['y'],borrow=True)
		valY=T.cast(valY,'int32')
		testX=theano.shared(testSet['x'],borrow=True)

		trainLabel=np.asarray(trainSet['y'],dtype=int)
		valLabel=np.asarray(valSet['y'],dtype=int)

		startTime=time.time()
		valTimeUsed=0

		index=T.iscalar('index')
		trainModel=theano.function(
			[index],self.cost,updates=self.adadeltaUpdate,
			givens={
			self.x:trainX[index*self.batchSize:(index+1)*self.batchSize],
			self.y:trainY[index*self.batchSize:(index+1)*self.batchSize]})
		print 'training model constructed!'

		runTrain=theano.function(
			[index],[self.errors,self.predict],
			givens={
			self.x:trainX[index*self.batchSize:(index+1)*self.batchSize],
			self.y:trainY[index*self.batchSize:(index+1)*self.batchSize]})
		print 'model to run train constructed!'

		runValidation=theano.function(
			[index],[self.errors,self.predict],
			givens={
			self.x:valX[index*self.batchSize:(index+1)*self.batchSize],
			self.y:valY[index*self.batchSize:(index+1)*self.batchSize]})
		print 'model to run validation constructed!'

		predictTest=theano.function(
			[],[self.predict],
			givens={self.x:testX})

		epoch=0
		maxValAcc4C=0.0
		maxValAcc2C=0.0
		maxValBAcc4C=0.0
		maxValBAcc2C=0.0
		Test2CResult=[]
		Test4CResult=[]

		while epoch<nEpoch:
			epoch+=1
			num=0
			for minBatch in np.random.permutation(range(trainBatches)):
				cost=trainModel(minBatch)

				if num%50==0:
					valStartTime=time.time()
					trainError=[]
					trainPredictLabel=[]
					for i in xrange(trainBatches):
						batchError,batchLabel=runTrain(i)
						trainError=trainError+[batchError]
						trainPredictLabel=trainPredictLabel+[batchLabel]
					trainError=np.asarray(trainError,dtype=float).reshape([trainBatches])
					trainPredictLabel=np.asarray(trainPredictLabel,dtype=int).reshape([trainSize])
					trainAcc=1-np.mean(trainError);
					assert(len(trainPredictLabel)==len(trainLabel))
					trainAccInfo=np.zeros([self.categories,self.categories],dtype=int)
					for i in xrange(trainSize):
						trainAccInfo[trainLabel[i],trainPredictLabel[i]]+=1
					trainAccs=np.zeros([self.categories],dtype=float)
					labelNum=trainAccInfo.sum(1)
					for i in xrange(self.categories):
						trainAccs[i]=float(trainAccInfo[i,i])/float(labelNum[i])
					trainBAcc4C=np.mean(trainAccs)

					valError=[]
					valPredictLabel=[]
					for i in xrange(valBatches):
						batchError,batchLabel=runValidation(i)
						valError=valError+[batchError]
						valPredictLabel=valPredictLabel+[batchLabel]
					valError=np.asarray(valError,dtype=float).reshape([valBatches])
					valPredictLabel=np.asarray(valPredictLabel,dtype=int).reshape([valSize])
					valAcc4C=1-np.mean(valError)
					assert(len(valLabel)==len(valPredictLabel))
					valAccInfo=np.zeros([self.categories,self.categories],dtype=int)
					for i in xrange(valSize):
						valAccInfo[valLabel[i],valPredictLabel[i]]+=1
					valAccs=np.zeros([self.categories],dtype=float)
					labelNum=valAccInfo.sum(1)
					for i in xrange(self.categories):
						valAccs[i]=float(valAccInfo[i,i])/float(labelNum[i])
					valBAcc4C=np.mean(valAccs)

					valAcc2C=float(np.sum(valAccInfo[:-1,:-1])+valAccInfo[-1,-1])/float(np.sum(valAccInfo))
					valBAcc2C=(float(np.sum(valAccInfo[:-1,:-1]))/float(np.sum(valAccInfo[:-1,:])) \
						+float(valAccInfo[-1,-1])/float(np.sum(valAccInfo[-1,:])))/2

					updateBAcc2C=valBAcc2C>maxValBAcc2C
					updateBAcc4C=valBAcc4C>maxValBAcc4C						
					maxValAcc4C=max(maxValAcc4C,valAcc4C)
					maxValBAcc4C=max(maxValBAcc4C,valBAcc4C)
					maxValAcc2C=max(maxValAcc2C,valAcc2C)
					maxValBAcc2C=max(maxValBAcc2C,valBAcc2C)
					if updateBAcc2C or updateBAcc4C:
						prediction=predictTest()
						if updateBAcc4C:
							Test4CResult=np.asarray(prediction,dtype=int)
						if updateBAcc2C:
							Test2CResult=(np.asarray(prediction,dtype=int)+1)/self.categories

					print '---------------outline----------------'
					print 'epoch=%i,num=%i:'%(epoch,num)
					print 'train precision=%.2f%%'%(trainAcc*100.)
					print 'validation precision=%.2f%%, best=%.2f%%'%(valAcc4C*100.,maxValAcc4C*100.)
					print 'binary validation precision=%.2f%%, best=%.2f%%'%(valAcc2C*100.,maxValAcc2C*100.)
					print 'train recall=%.2f%%'%(trainBAcc4C*100.)
					print 'validation recall=%.2f%%, best=%.2f%%'% \
					(valBAcc4C*100.,maxValBAcc4C*100.)
					print 'binary validation recall=%.2f%%, best=%.2f%%'% \
					(valBAcc2C*100.,maxValBAcc2C*100.)
					print '---------------details----------------'
					print 'train:'
					for i in xrange(self.categories):
						for j in xrange(self.categories):
							print trainAccInfo[i,j],'\t',
						print ''
					print 'validation:'
					for i in xrange(self.categories):
						for j in xrange(self.categories):
							print valAccInfo[i,j],'\t',
						print ''
					valEndTime=time.time()
					valTimeUsed+=valEndTime-valStartTime
				num+=1

		endTime=time.time()
		timeUsed=endTime-startTime-valTimeUsed
		print 'Time Consumed: %f second'%timeUsed

		return [maxValAcc4C,Test4CResult,Test2CResult]
