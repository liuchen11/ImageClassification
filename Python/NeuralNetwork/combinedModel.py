import sys
import cPickle
import time
import theano

import numpy as np
import theano.tensor as T

from collections import defaultdict, OrderedDict
from convLayer import *
from hiddenLayer import *
from logisticRegression import *
from loadImg import *

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

class CombinedModel(object):

	def __init__(self,mlpShape,mlpNeurons,mlpDropout,
				cnnShape,cnnFeatures,cnnFilters,cnnPoolings,cnnDropout,
				wdecay,categories,learningRate,name):
		'''
		>>>mlp-cnn combined model

		>>>type mlpShape:list [batchSize,inputNeurons]
		>>>type cnnShape:list [batchSize,inputFeatures,width,height]
		>>>para mlpShape/cnnSHape: inputs' shapes of mlp/cnn layers
		>>>type mlpNeurons:tuple or list of int
		>>>para mlpNeurons:number of neurons in each mlp layer
		>>>type cnnFeatures:tuple or list of int
		>>>para cnnFeatures:number of feaures in each cnn layer
		>>>type cnnFilters/cnnPoolings:tuple or list of (int,int)
		>>>para cnnFilters/cnnPoolings:shapes of filters and poolings in each cnn layer
		>>>type mlpDropout/cnnDropout:tuple or list of int
		>>>para mlpDropout/cnnDropout:dropout rate of each mlp/cnn layer
		>>>type wdecay:float
		>>>para wdecay:weight decay
		>>>type categories:int
		>>>para categories:number of output categories
		>>>type learningRate:float
		>>>para learningRate:learning rate
		>>>type name:str
		>>>para name:model's name
		'''
		self.name=name
		self.categories=categories
		self.learningRate=learningRate
		rng=np.random.RandomState(254860)

		assert(mlpShape[0]==cnnShape[0])
		self.batchSize,mlpInputSize=mlpShape
		self.batchSize,cnnInputFeatures,width,height=cnnShape

		self.mlpX=T.matrix('mlp')
		self.cnnX=T.tensor4('cnn')
		self.y=T.ivector('y')
		self.mlpLayers=[]
		self.cnnLayers=[]
		self.fcLayers=[]

		##MLP Part##
		mlpLayerNum=min(len(mlpNeurons),len(mlpDropout))
		print 'mlpLayers: %i'%mlpLayerNum

		mlpCurrentInput=self.mlpX
		mlpCurrentInNum=mlpInputSize
		mlpCurrentOutNum=mlpNeurons[0]

		for i in xrange(mlpLayerNum):
			currentLayer=DropoutHiddenLayer(
				rng=rng,
				input=mlpCurrentInput,
				n_in=mlpCurrentInNum,
				n_out=mlpCurrentOutNum,
				activation=ReLU,
				dropoutRate=mlpDropout[i]
				)
			self.mlpLayers.append(currentLayer)
			mlpCurrentInput=currentLayer.output
			mlpCurrentInNum=mlpNeurons[i]
			if i!=mlpLayerNum-1:
				mlpCurrentOutNum=mlpNeurons[i+1]

		##CNN Part##
		cnnLayerNum=min(len(cnnFilters),len(cnnFeatures),len(cnnPoolings),len(cnnDropout))
		print 'cnnLayers: %i'%cnnLayerNum

		cnnCurrentInput=self.cnnX
		cnnCurrentShape=cnnShape
		cnnCurrentFilters=[cnnFeatures[0],cnnInputFeatures,cnnFilters[0][0],cnnFilters[0][1]]
		cnnCurrentPoolings=cnnPoolings[0]
		cnnCurrentInNum=0
		cnnCurrentOutNum=0
		flatten=False

		for i in xrange(cnnLayerNum):
			if flatten==False and (cnnCurrentFilters[2]<=0 or cnnCurrentFilters[3]<=0):
				cnnCurrentInput=cnnCurrentInput.flatten(2)
				cnnCurrentInNum=np.prod(cnnCurrentShape[1:])
				cnnCurrentOutNum=cnnFeatures[i]
				flatten=True
			if flatten==False:
				currentLayer=DropoutConvLayer(
					rng=rng,
					input=cnnCurrentInput,
					shape=cnnCurrentShape,
					filters=cnnCurrentFilters,
					pool=cnnCurrentPoolings,
					dropoutRate=cnnDropout[i]
					)
				self.cnnLayers.append(currentLayer)
				cnnCurrentInput=currentLayer.output
				currentWidth=(cnnCurrentShape[2]-cnnCurrentFilters[2]+1)/cnnCurrentPoolings[0]
				currentHeight=(cnnCurrentShape[3]-cnnCurrentFilters[3]+1)/cnnCurrentPoolings[1]
				cnnCurrentShape=[cnnCurrentShape[0],cnnFeatures[i],currentWidth,currentHeight]
				if i!=cnnLayerNum-1:
					cnnCurrentFilters=[cnnFeatures[i+1],cnnFeatures[i],cnnFilters[i+1][0],cnnFilters[i+1][1]]
					cnnCurrentPoolings=cnnPoolings[i+1]
			else:
				currentLayer=DropoutHiddenLayer(
					rng=rng,
					input=cnnCurrentInput,
					n_in=cnnCurrentInNum,
					n_out=cnnCurrentOutNum,
					activation=ReLU,
					dropoutRate=cnnDropout[i]
					)
				self.fcLayers.append(currentLayer)
				cnnCurrentInput=currentLayer.output
				cnnCurrentInNum=cnnFeatures[i]
				if i!=cnnLayerNum-1:
					cnnCurrentOutNum=feaures[i+1]

		if flatten==False:
			cnnCurrentInput=cnnCurrentInput.flatten(2)
			cnnCurrentInNum=np.prod(cnnCurrentShape[1:])

		##Classifier##
		self.classifier=LogisticRegression(
			input=T.concatenate([mlpCurrentInput,cnnCurrentInput],axis=1),
			n_in=cnnCurrentInNum+mlpCurrentInNum,
			n_out=self.categories
			)
		weights=0
		for param in self.classifier.param:
			weights+=T.sum(T.sqr(param))

		self.params=self.classifier.param

		for subNets in self.mlpLayers:
			self.params+=subNets.param
		for subNets in self.cnnLayers:
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

		>>>trainSet/valSet: training/validation set
		>>>nEpoch: maximum iterations
		'''
		trainSize=trainSet['y'].shape[0]
		valSize=valSet['y'].shape[0]
		testSize=testSet['cnnX'].shape[0]
		trainBatches=trainSize/self.batchSize
		valBatches=valSize/self.batchSize

		trainMlpX=theano.shared(trainSet['mlpX'],borrow=True)
		trainCnnX=theano.shared(trainSet['cnnX'],borrow=True)
		trainY=theano.shared(trainSet['y'],borrow=True)
		trainY=T.cast(trainY,'int32')
		valMlpX=theano.shared(valSet['mlpX'],borrow=True)
		valCnnX=theano.shared(valSet['cnnX'],borrow=True)
		valY=theano.shared(valSet['y'],borrow=True)
		valY=T.cast(valY,'int32')
		testMlpX=testSet['mlpX']
		testCnnX=testSet['cnnX']

		trainLabel=np.asarray(trainSet['y'],dtype=int)
		valLabel=np.asarray(valSet['y'],dtype=int)

		startTime=time.time()
		valTimeUsed=0

		index=T.iscalar('index')
		trainModel=theano.function(
			[index],self.cost,updates=self.adadeltaUpdate,
			givens={
			self.mlpX:trainMlpX[index*self.batchSize:(index+1)*self.batchSize],
			self.cnnX:trainCnnX[index*self.batchSize:(index+1)*self.batchSize],
			self.y:trainY[index*self.batchSize:(index+1)*self.batchSize]})
		print 'training model constructed!'

		runTrain=theano.function(
			[index],[self.errors,self.predict],
			givens={
			self.mlpX:trainMlpX[index*self.batchSize:(index+1)*self.batchSize],
			self.cnnX:trainCnnX[index*self.batchSize:(index+1)*self.batchSize],
			self.y:trainY[index*self.batchSize:(index+1)*self.batchSize]})
		print 'model to run train constructed!'

		runValidation=theano.function(
			[index],[self.errors,self.predict],
			givens={
			self.mlpX:valMlpX[index*self.batchSize:(index+1)*self.batchSize],
			self.cnnX:valCnnX[index*self.batchSize:(index+1)*self.batchSize],
			self.y:valY[index*self.batchSize:(index+1)*self.batchSize]})
		print 'model to run validation constructed!'

		testMlpInput=T.matrix('testMlpInput')
		testCnnInput=T.tensor4('testCnnInput')
		mlpCurrentInput=testMlpInput
		cnnCurrentInput=testCnnInput
		for layers in self.mlpLayers:
			mlpCurrentInput=layers.process(mlpCurrentInput)
		for layers in self.cnnLayers:
			cnnCurrentInput=layers.process(cnnCurrentInput,testSize)
		cnnCurrentInput=cnnCurrentInput.flatten(2)
		for layers in self.fcLayers:
			cnnCurrentInput=layers.process(cnnCurrentInput)
		testPrediction=self.classifier.predictInstance(\
			T.concatenate([mlpCurrentInput,cnnCurrentInput],axis=1))
		predictTest=theano.function([testMlpInput,testCnnInput],[testPrediction])
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
						prediction=predictTest(testMlpX,testCnnX)
						if updateBAcc4C:
							Test4CResult=np.asarray(prediction,dtype=int)
						if updateBAcc2C:
							Test2CResult=(np.asarray(prediction,dtype=int)+1)/self.categories


					print '---------------outline----------------'
					print 'epoch=%i,num=%i:'%(epoch,num)
					print 'train precision=%f%%'%(trainAcc*100.)
					print 'validation precision=%f%%, best=%f%%'%(valAcc4C*100.,maxValAcc4C*100.)
					print 'binary validation precision=%f%%, best=%f%%'%(valAcc2C*100.,maxValAcc2C*100.)
					print 'balanced train precision=%f%%'%(trainBAcc4C*100.)
					print 'balanced validation precision=%f%%, best=%f%%'% \
					(valBAcc4C*100.,maxValBAcc4C*100.)
					print 'balanced binary validation precision=%f%%, best=%f%%'% \
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


