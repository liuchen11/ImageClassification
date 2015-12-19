import numpy as np

import theano
import theano.tensor as T

class HiddenLayer(object):
	def __init__(self,rng,input,n_in,n_out,activation):
		'''
		>>>type rng: numpy.random.RandomState
		>>>para rng: initalize weight randomly
		>>>type input: theano.tensor.matrix
		>>>para input: input data
		>>>type n_in: int
		>>>para n_in: the num of input neurons
		
		>>>type n_out: int
		>>>para n_out: the num of output neurons
		>>>type activation: func
		>>>para activation: the activate function
		'''
		self.input=input
		self.activation=activation

		w_bound=np.sqrt(6.0/(n_in+n_out))

		w_value=np.asarray(
			rng.uniform(low=-w_bound,high=w_bound,size=(n_in,n_out)),
			dtype=theano.config.floatX
			)
		
		if activation==T.nnet.sigmoid:
			w_value*=4
		self.w=theano.shared(value=w_value,name='w',borrow=True)

		b_value=np.zeros((n_out),dtype=theano.config.floatX)
		self.b=theano.shared(value=b_value,name='b',borrow=True)

		raw_output=T.dot(input,self.w)+self.b

		self.output=(
			raw_output if activation is None
			else activation(raw_output)
			)

		self.param=[self.w,self.b]

	def process(self,data):
		'''
		>>>process newly input data
		>>>type data: T.matrix
		>>>para data: newly input data
		'''
		raw_output=T.dot(data,self.w)+self.b

		output=(
			raw_output if self.activation is None
			else self.activation(raw_output)
			)

		return output

def dropout(rng,value,p):
	'''
	>>>dropout function
	>>>type rng: np.random.RandomState
	>>>para rng: random seed
	>>>type value: T.tensor4
	>>>para value: input value
	>>>type p: float
	>>>para p: dropout rate
	'''
	srng=T.shared_randomstreams.RandomStreams(rng.randint(254860))
	mask=srng.binomial(n=1,p=1-p,size=value.shape)
	return value*T.cast(mask,theano.config.floatX)

class DropoutHiddenLayer(HiddenLayer):
	
	def __init__(self,rng,input,n_in,n_out,activation,dropoutRate):
		HiddenLayer.__init__(self,rng,input,n_in,n_out,activation)
		self.dropoutRate=dropoutRate
		self.output=dropout(rng,self.output,dropoutRate)

	def process(self,data):
		output=HiddenLayer.process(self,data)
		return output*(1-self.dropoutRate)