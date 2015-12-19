import sys
sys.path.append('../')
import numpy as np

from loadImg import *
from kohonen import *

if __name__=='__main__':
	if len(sys.argv)!=2 or (sys.argv[1]!='-hog' and sys.argv[1]!='-overfeat'):
		print 'Usage: python main.py [-hog | -overfeat]'
		exit(0)

	featureType=sys.argv[1]
	featureNum=5408 if sys.argv[1]=='-hog' else 36864

	[cnnFeature,hogFeature,labels]=getDataY()
	assert(cnnFeature.shape[0]==labels.shape[0])
	assert(hogFeature.shape[0]==labels.shape[0])

	instanceNum=labels.shape[0]
	trainInstanceNum=int(instanceNum*0.8)
	randIndex=np.random.permutation(range(instanceNum))

	trainSetX=hogFeature[randIndex[:trainInstanceNum]] if featureType=='-hog' \
				else cnnFeature[randIndex[:trainInstanceNum],:-1]
	trainSetY=labels[randIndex[:trainInstanceNum]]
	validateSetX=hogFeature[randIndex[trainInstanceNum:]] if featureType=='-hog' \
				else cnnFeature[randIndex[trainInstanceNum:],:-1]
	validateSetY=labels[randIndex[trainInstanceNum:]]

	trainSetX=np.asarray(trainSetX,dtype=float).reshape(trainInstanceNum,featureNum)
	trainSetY=np.asarray(trainSetY,dtype=float).reshape(trainInstanceNum)-1
	validateSetX=np.asarray(validateSetX,dtype=float).reshape(instanceNum-trainInstanceNum,featureNum)
	validateSetY=np.asarray(validateSetY,dtype=float).reshape(instanceNum-trainInstanceNum)-1

	model=kohonen(
		sizeK=8,
		dims=featureNum,
		classes=4,
		eta=0.05,
		sigma=2,
		vanishing=True
		)
	model.training(trainSetX,trainSetY,validateSetX,validateSetY,25)
