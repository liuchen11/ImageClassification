import os
import Image
import numpy as np

from math import *
from scipy.io import loadmat

imgFolder=os.path.dirname(os.path.realpath(__file__))+'/../data/imgs/'
trainFeatureFile=os.path.dirname(os.path.realpath(__file__))+'/../data/train.mat'
testFeatureFile=os.path.dirname(os.path.realpath(__file__))+'/../data/test.mat'

def getDataX(begin,end):
	'''
	>>>get the pixel information from images
	>>>output format: [imageIndex,Channel,Height,Width]
	'''
	imageNum=end-begin+1
	data=np.zeros([imageNum,3,231,231],dtype=np.float32)
	for index in xrange(imageNum):
		data[index]=loadImagePixelInfo(index+begin)
	return data

def getDataY():
	'''
	>>>get the label and pre-extracted feature from *.mat file
	>>>output format: [cnn_feature, hog_feature, labels]
	'''
	features=loadmat(trainFeatureFile)
	cnnFeature=features['train'][0][0][0]
	hogFeature=features['train'][0][0][1]
	labels=features['train'][0][0][2]
	return [cnnFeature, hogFeature, labels]

def getTestData():
	'''
	>>>get the features from test dataset
	>>>output format: [cnn_feature, hog_feature]
	'''
	features=loadmat(testFeatureFile)
	cnnFeature=features['test'][0][0][0]
	hogFeature=features['test'][0][0][1]
	return [cnnFeature,hogFeature]

def loadImagePixelInfo(index):
	'''
	>>>get the pixel information from one image

	>>>type index:int
	>>>para index:image number
	'''
	digitNum=int(log10(index))
	imgName=imgFolder+'train'+'0'*(4-digitNum)+'.jpg'
	imgFile=Image.open(imgName)
	pix=imgFile.load()

	try:
		assert(imgFile.size==(231,231))
	except Exception, e:
		print 'Image '+str(index)+': wrong size!'

	pixels=np.zeros([3,231,231],dtype=np.float32)
	if type(pix[0,0])==tuple and len(pix[0,0])==3: #color
		for i in xrange(231):
			for j in xrange(231):
				pixels[0,i,j]=pix[i,j][0]
				pixels[1,i,j]=pix[i,j][1]
				pixels[2,i,j]=pix[i,j][2]
	elif type(pix[0,0])==int:  #black&white
		for i in xrange(231):
			for j in xrange(231):
				pixels[0,i,j]=pix[i,j]
				pixels[1,i,j]=pix[i,j]
				pixels[2,i,j]=pix[i,j]
	else:
		print 'Image '+str(index)+': unparsable!'

	#normalization
	pixels[0]=(pixels[0]-pixels[0].mean())/pixels[0].std()
	pixels[1]=(pixels[1]-pixels[1].mean())/pixels[1].std()
	pixels[2]=(pixels[2]-pixels[2].mean())/pixels[2].std()
	return pixels