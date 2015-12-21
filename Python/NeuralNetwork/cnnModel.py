import sys
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

class CNNModel(object):

	def __init__(self,shape,filters,poolings,features,wdecay,
			categories,dropoutRate,learningRate,name):
		'''
		>>>initalize the model

		>>>type shape: tuple or list of length 4
		>>>para shape: [batchSize,channels,width,height]
		>>>type filters: tuple or list of (f1,f2)
		>>>para filters: sizes of filters in different layers
		>>>type poolings: tuple of list of (p1,p2)
		>>>para poolings: poolings operators's sizes in different layers
		>>>type features: tuple or list of int
		>>>para features: num of feature maps in each layer
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
		self.learningRate=learningRate
		self.name=name
		self.categories=categories
		rng=np.random.RandomState(254860)

		self.batchSize,self.channels,self.width,self.height=shape
		convLayerNum=min(len(filters),len(features),len(poolings),len(dropoutRate))

		self.x=T.tensor4('x')
		self.y=T.ivector('y')
		self.convLayers=[]
		self.fcLayers=[]

		currentShape=shape
		currentInput=self.x
		currentFilters=[features[0],self.channels,filters[0][0],filters[0][1]]
		currentPoolings=poolings[0]
		currentInNum=0
		currentOutNum=0
		flatten=False

		for i in xrange(convLayerNum):
			if flatten==False:
				currentLayer=DropoutConvLayer(
					rng=rng,
					input=currentInput,
					shape=currentShape,
					filters=currentFilters,
					pool=currentPoolings,
					dropoutRate=dropoutRate[i]
					)
				self.convLayers.append(currentLayer)
				currentWidth=(currentShape[2]-currentFilters[2]+1)/currentPoolings[0]
				currentHeight=(currentShape[3]-currentFilters[3]+1)/currentPoolings[1]
				currentShape=[currentShape[0],features[i],currentWidth,currentHeight]
				currentInput=currentLayer.output
				if i!=convLayerNum-1:
					currentFilters=[features[i+1],features[i],filters[i+1][0],filters[i+1][1]]
					currentPoolings=poolings[i+1]
					if currentFilters[2]<=0 or currentFilters[3]<=0:
						currentInput=currentInput.flatten(2)
						currentInNum=np.prod(currentShape[1:])
						currentOutNum=features[i+1]
						flatten=True
			else:
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
				currentInNum=features[i]
				if i!=convLayerNum-1:
					currentOutNum=features[i+1]

		if flatten==False:
			currentInput=currentInput.flatten(2)
			currentInNum=np.prod(currentShape[1:])

		self.classifier=LogisticRegression(
			input=currentInput,
			n_in=currentInNum,
			n_out=self.categories
			)

		weights=0
		for param in self.classifier.param:
			weights+=T.sum(T.sqr(param))

		self.params=self.classifier.param
		for subNets in self.convLayers:
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

		print 'the model '+self.name+' constructed!'

	def training(self,trainSet,valSet,testSet,nEpoch):
		'''
		>>>training the model

		>>>trainSet/valSet: training/validating set
		>>>nEpoch: maximum iterations
		'''
		trainSize=trainSet['x'].shape[0]
		valSize=valSet['x'].shape[0]
		testSize=testSet['x'].shape[0]
		trainBatches=trainSize/self.batchSize
		valBatches=valSize/self.batchSize

		trainX=theano.shared(trainSet['x'],borrow=True)
		trainY=theano.shared(trainSet['y'],borrow=True)
		trainY=T.cast(trainY,'int32')
		valX=theano.shared(valSet['x'],borrow=True)
		valY=theano.shared(valSet['y'],borrow=True)
		valY=T.cast(valY,'int32')
		testX=testSet['x']

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

		testInput=T.tensor4('testX')
		currentInput=testInput
		for layer in self.convLayers:
			currentInput=layer.process(currentInput,testSize)
		currentInput=currentInput.flatten(2)
		for layer in self.fcLayers:
			currentInput=layer.process(currentInput)
		testPrediction=self.classifier.predictInstance(currentInput)
		predictTest=theano.function([testInput],[testPrediction])
		print 'model to predict labels on testing set constructed!'

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
						prediction=predictTest(testX)
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
