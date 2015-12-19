import numpy as np

from sklearn import svm

class SVM(object):

	def __init__(self,nClass,kernel):

		'''
		>>>construct a SVM classifier

		>>>type nClass: int
		>>>para nClass: the number of classes
		>>>type kernel: str/function
		>>>para kernel: specify or customize the kernel function
		'''
		self.classifier=svm.SVC(kernel=kernel)
		self.classifier.decision_function_shape='ovr'
		self.nClass=nClass

	def training(self,trainX,trainY,validateX,validateY):
		'''
		>>>train a SVM

		>>>type dataX: matrix of shape [instances, features]
		>>>para dataX: training/validating data
		>>>type dataY: vector of shape [instances]
		>>>para dataY: labels
		'''
		self.classifier.fit(trainX,trainY)
		print '-------trainSet--------'
		self.test(trainX,trainY)
		print '------validateSet------'
		self.test(validateX,validateY)

	def test(self,dataX,dataY):
		'''
		>>>test a benchmark
		'''
		predictY=self.classifier.predict(dataX)

		accMatrix=np.zeros([self.nClass,self.nClass],dtype=int)
		for i,predictLabel in enumerate(predictY):
			realLabel=dataY[i]
			accMatrix[realLabel,predictLabel]+=1

		instanceNum=predictY.shape[0]
		subAcc=np.zeros([self.nClass],dtype=float)
		corrects=0
		for i in xrange(self.nClass):
			subAcc[i]=float(accMatrix[i,i])/float(np.sum(accMatrix[i]))
			corrects+=accMatrix[i,i]

		unbalanced=float(corrects)/float(instanceNum)
		balanced=np.mean(subAcc)

		print '-------result-------'
		print 'unbalanced: %f%%, balanced: %f%%'%(unbalanced*100.,balanced*100.)
		print '-------details------'
		for i in xrange(self.nClass):
			for j in xrange(self.nClass):
				print accMatrix[i,j],'\t',
			print ''
