import numpy as np

def gauss(x,para):
	'''
	>>>Gaussian distribution
	'''
	return np.exp(-(x-para[0])**2/(2*para[1]**2))

class kohonen(object):

	def __init__(self,sizeK,dims,classes,eta,sigma,vanishing):
		'''
		>>>construct a kohonen neural network

		>>>type sizeK: int
		>>>para sizeK: the size of neurons
		>>>type dims: int
		>>>para dims: the number of dims of features
		>>>type classes: int
		>>>para classes: the number of categories
		>>>type eta: float
		>>>para eta: learning rate
		>>>type sigma: float
		>>>para sigma: initial sigma in neighbourhood function
		>>>type vanishing: bool
		>>>para vanishing: whether or not to use vanishing sigma
		'''
		self.sizeK=sizeK
		self.dims=dims
		self.sigma=float(sigma)
		self.eta=float(eta)
		self.vanishing=vanishing
		self.dataSize=0
		self.categories=classes
		self.classMatrix=np.zeros([self.sizeK**2,classes],dtype=int)

		self.centers=np.random.rand(self.sizeK**2,dims)
		self.neighbour=np.arange(sizeK**2).reshape((sizeK,sizeK))

	def update(self,instance,label):
		'''
		>>>update the weight of the network when a new data comes

		>>>type instance: numpy.array
		>>>para instance: the newly-come data
		>>>type label: int
		>>>para label: the label of this instance
		'''
		loc=np.argmin(np.sum((self.centers-np.resize(instance,
			(self.sizeK**2,instance.shape[0])))**2,1))
		x0,y0=np.nonzero(self.neighbour==loc)
		self.dataSize+=1
		self.classMatrix[loc,label]+=1

		for j in xrange(self.sizeK**2):
			x1,y1=np.nonzero(self.neighbour==j)
			disc=gauss(np.sqrt((x0-x1)**2+(y0-y1)**2),[0,self.sigma])
			self.centers[j,:]+=self.eta*disc*(instance-self.centers[j,:])

	def training(self,trainX,trainY,validateX,validateY,nEpoch):
		'''
		>>>training a kohonen neural network

		>>>type dataX: matrx of shape [instanceNum, features]
		>>>para dataX: training/validating data
		>>>type dataY: vector of int
		>>>para dataY: labels
		'''
		trainInstanceNum,features=trainX.shape
		validateInstanceNum,features=validateX.shape
		assert(features==self.dims)
		assert(trainX.shape[0]<=trainY.shape[0])
		assert(validateX.shape[0]<=validateY.shape[0])

		for i in xrange(nEpoch):
			order=np.arange(trainInstanceNum)
			np.random.shuffle(order)
			dataX=trainX[order]
			dataY=trainY[order]

			for j in xrange(trainInstanceNum):
				self.update(dataX[j],dataY[j])

			print 'Epoch=%i,sizeK=%i'%(i,self.sizeK)
			print 'eta=%f,sigma=%f,'%(self.eta,self.sigma)
			print '------Training Set--------'
			self.test(trainX,trainY)
			print '------Testing Set---------'
			self.test(validateX,validateY)
			if self.vanishing:
				self.sigma/=2
		

	def test(self,dataX,dataY):
		'''
		>>>test a batch of data and return the (balanced) accuracy

		>>>type dataX: matrx of shape [instanceNum, features]
		>>>para dataX: test data
		>>>type dataY: vector of int
		>>>para dataY: labels
		'''
		instanceNum, features=dataX.shape
		assert(features==self.dims)
		assert(dataX.shape[0]<=dataY.shape[0])
		'''
		results[i,j] shows the number of instances that belongs to class i
		and are classified as class j
		'''
		results=np.zeros([self.categories,self.categories],dtype=int)

		for i in xrange(instanceNum):
			instance=dataX[i]
			loc=np.argmin(np.sum((self.centers-np.resize(instance,
				(self.sizeK**2,instance.shape[0])))**2,1))
			predictLabel=np.argmax(self.classMatrix[loc])
			realLabel=dataY[i]
			results[realLabel,predictLabel]+=1

		subAcc=np.zeros([self.categories,],dtype=float)
		rights=0
		for i in xrange(self.categories):
			subAcc[i]=float(results[i,i])/float(np.sum(results[i]))
			rights+=results[i,i]
		balanced=np.mean(subAcc)
		unbalanced=float(rights)/float(instanceNum)
		print 'Accuracy: unbalanced %f%%, balanced %f%%'%(unbalanced*100.,balanced*100.)
		for i in xrange(self.categories):
			for j in xrange(self.categories):
				print results[i,j],'\t',
			print ''
		return unbalanced,balanced




