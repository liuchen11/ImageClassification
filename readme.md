#Instructions#
We implement many kinds of classifiers on image classification in python and matlab. The images are firstly preprocessed and embedded into high-dimensional vectors. Our classifiers are all based on these vectors and have nothing to do with feature extraction from images.

The outline of files and directories is below. In both training and testing set, two kinds of features (Overfeat and HoG) are provided. **Running these classifiers may need up to 4-5GB RAM, please make sure your machine has enough memory.**

```
|-Python
	|-NeuralNetwork     #NeuralNetwork Models, such as CNN and MLP
	|-SVM			    #Support Vector Machine
	|-SOM			    #Self-Organized Map
	|-loadImg.py	    #Data Loader
|-Matlab
    |-trainTestTradeOff #deciding the size of training and validation data sets
    |-Logistic	        #Logistic regression
    |-SVM	            #Support vector machine
|-data
	|-train.mat  		#Training Set: Data and Label
	|-test.mat  		#Testing Set: Data
|-results				#Output Results in *.mat Format
```

##1.Python##
In part of python implementation, we use several dependent libraries: **numpy**, **theano**, **sklearn** and **scipy**. To parse source image, we use library **Image**. In each subdirectory, there is an identical files **loadImg.py** loading feature vectors from images.

###1.1 Neural Networks###
* The files in this part:

```
|-NeuralNetwork
	|-hiddenLayer.py		#Hidden Layer
	|-convLayer.py			#Convolutional Layer(including pooling)
	|-logisticRegression.py	#Softmax Classifier
	|-mlpModel.py			#Multi-Layer Perceptron
	|-cnnModel.py			#Convolutional Neural Network
	|-combinedModel.py		#Model Connecting MLP and CNN in Parallel
	|-overfeatMain.py		#Run Nets on Overfeat Features
	|-hogMain.py			#Run Nets on HoG Features
	|-combinedMain.py		#Run Combined Model
```
* The output pattern:

```
...
---------------outline----------------
epoch=5,num=100:
train precision=99.98%
validation precision=90.00%, best=90.58%
binary validation precision=90.42%, best=91.25%
train recall=99.97%
validation recall=90.23%, best=91.90%
binary validation recall=90.30%, best=90.89%
---------------details----------------
train:
748 	0 		0 		1
0 		933 	0 		0
0 		0 		1179 	0
0 		0 		0 		1939
validation:
193 	1 		1 		20
2 		214 	1 		12
0 		0 		275 	38
11 		2 		32 		398
---------------outline----------------
epoch=5,num=150:
train precision=100.00%
validation precision=89.83%, best=90.58%
binary validation precision=90.50%, best=91.25%
train recall=100.00%
validation recall=90.54%, best=91.90%
binary validation recall=89.90%, best=90.89%
---------------details----------------
train:
749 	0 		0 		0
0 		933 	0 		0
0 		0 		1179 	0
0 		0 		0 		1939
validation:
198 	1 		0 		16
4 		216 	1 		8
2 		0 		276 	35
14 		7 		34 		388
Time Consumed: 102.985855 second
```
* **The code of this part can be run much faster on GPU by adding the following part before your command shown in the following bullets**. You should replace ``gpu1`` with your GPU name.

```
THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python <file> <command>
Example:
THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python hogMain.py -cnn
```
* Run MLP on Overfeat Features

```
python overfeatMain.py (-binary)
```
* Run MLP/CNN on HoG Features

```
python hogMain.py -mlp (-binary)
python hogMain.py -cnn (-binary)
```
* Run Combined Model

```
python combinedMain.py (-binary)
```
###1.2 Self-Organized Map###
* The files in this part:

```
|-SOM
	|-kohonen.py		#SOM
	|-main.py			#Main Function
```
* Output pattern:

```
...
Epoch=3,sizeK=8
eta=0.050000,sigma=0.2500,
------Training Set--------
Accuracy: unbalanced 62.96%, balanced 62.08%
428 	182 	27 		126
83 		703 	38 		105
8 		46 		564 	578
31 		148 	406 	1327
------Testing Set---------
Accuracy: unbalanced 60.58%, balanced 60.40%
109 	49 		11 		32
24 		179 	6 		24
2 		3 		137 	154
4 		49 		115 	302
Epoch=4,sizeK=8
eta=0.050000,sigma=0.125000,
------Training Set--------
Accuracy: unbalanced 64.02%, balanced 62.48%
462 	179 	24 		98
119 	686 	34 		90
9 		48 		474 	665
47 		139 	275 	1451
------Testing Set---------
Accuracy: unbalanced 61.00%, balanced 60.41%
115 	49 		11 		26
31 		175 	5 		22
2 		4 		122 	168
11 		47 		92 		320
```
* Run SOM in a supervised manner on hog or overfeat feature

```
python main.py -hog
python main.py -overfeat
```

###1.3 SVM###
* The files in this part:

```
|-SVM
	|-SVM.py			#SVM
	|-main.py 		#Main Function
```
* Output pattern

```
feature: -overfeat, kernel: linear
-------trainSet--------
-------result-------
unbalanced: 100.00%, balanced: 100.00%
-------details------
766 	0 		0 		0
0 		936 	0 		0
0 		0 		1208 	0
0 		0 		0 		1890
------validateSet------
-------result-------
unbalanced: 92.50%, balanced: 92.35%
-------details------
184 	1 		0 		13
2 		214 	0 		10
0 		0 		249 	35
4 		4 		21 		463
```
* Run SVM of different kernel in hog or overfeat feature

```
python main.py [-hog|-overfeat] [<kernel>]
Example:
python main.py -hog rbf
python main.py -overfeat linear
```
##2.MATLAB##
In part of MATLAB implementation, please place the training and testing data in the 'Matlab' directory, or modify the 'load' command within all codes. You can run the enclosed MATLAB scripts by pressing the start button.
