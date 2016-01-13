import numpy as np
import theano
import theano.tensor as T

class MLP(object):
    def __init__(self,n_in,n_hid,n_out,inp):
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.params = self.initialize_weights(self.n_in,self.n_hid,self.n_out)
        self.X = T.TensorType(dtype='float32', broadcastable=())('X')
        self.X = inp
        self.hidden =  T.nnet.relu(T.dot(self.W0,self.X.T) + self.b0.dimshuffle(0,'x'))
        self.hidden1 =  T.nnet.relu(T.dot(self.W2,self.hidden) + self.b2.dimshuffle(0,'x'))
        self.output = T.tanh((T.dot(self.W1,self.hidden1) + self.b1.dimshuffle(0,'x')).T)
        print self.output.dtype
        print "Gaussian MLP built"
    def initialize_weights(self,n_in,n_hid,n_out):
        self.W0 = theano.shared(np.random.normal(0,0.001,size=(n_hid,n_in)).astype('float32'),name="MLPW0")
        self.W1 = theano.shared(np.random.normal(0,0.001,size=(n_out,n_hid)).astype('float32'),name="MLPW1")
        self.W2 = theano.shared(np.random.normal(0,0.001,size=(n_hid,n_hid)).astype('float32'),name="MLPW2")
        self.b2 = theano.shared(np.random.normal(0,0.001,size=(n_hid,)).astype('float32'),name="MLPb2")
        self.b0 = theano.shared(np.random.normal(0,0.001,size=(n_hid,)).astype('float32'),name="MLPb0")
        self.b1 = theano.shared(np.random.normal(0,0.001,size=(n_out,)).astype('float32'),name="MLPb1")
        #self.W0 = theano.shared(np.random.normal(0,1,size=(n_hid,n_in)) ,name="MLPW0")
        #self.W1 = theano.shared(np.random.normal(0,1,size=(n_out,n_hid)) ,name="MLPW1")
        #self.W2 = theano.shared(np.random.normal(0,1,size=(n_hid,n_hid)) ,name="MLPW2")
        #self.b2 = theano.shared(np.zeros(shape=(n_hid,)) ,name="MLPb2")
        #self.b0 = theano.shared(np.zeros(shape=(n_hid,)) ,name="MLPb0")
        #self.b1 = theano.shared(np.zeros(shape=(n_out,)) ,name="MLPb1")
        return [self.W0,self.W1,self.W2,self.b0,self.b1,self.b2]
    
if __name__ =="__main__":
    print "Testing forward propagation"
    X = T.matrix("MLPX")
    mlp = MLP(n_in=20,n_hid=200,n_out=30,inp=X)
    entry = np.random.normal(0,1,size=(10,20))
    mlp_func = theano.function([],mlp.output,givens={X:entry})
    print mlp_func().shape
