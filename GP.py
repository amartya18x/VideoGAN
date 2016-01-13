import numpy as np
import theano
import theano.tensor as T
from theano.tensor import slinalg as sin
from theano.tensor.shared_randomstreams import RandomStreams
dtype = theano.config.floatX
class GP(object):
    def __init__(self,size=20,v0=1,v1=1,v2=1,r=1,time_step=10,alpha=2):
        self.lsize = size
        self.time = time_step
        self.dtype = theano.config.floatX
        self.v0 = theano.shared(np.cast[dtype](v0),name="v0")
        self.v1 = theano.shared(np.cast[dtype](v1),name="v1")
        self.v2 = theano.shared(np.cast[dtype](v2),name="v2")
        self.alpha = theano.shared(np.cast[dtype](alpha),name="alpha")
        self.gamma = theano.shared(np.cast[dtype](r),name="r")
        self.srng = RandomStreams(seed=234)
        self.DiffM = theano.shared(self.create_mat(self.time,self.lsize),name="diffM")
        self.params = [self.v0,self.gamma,self.alpha]
        self.sample = self.return_output(self.DiffM)
        
##This takes the difference matrix and samples from the GP and returns a signal
    def return_output(self,Dif):
        #Dif is theano.Tensor.matrix type
        Frac = Dif/self.gamma
        Cov = self.v0*T.pow(Frac,self.alpha)
        L = sin.cholesky(T.exp(-Cov))
        eps = self.srng.uniform((self.time,self.lsize))
        return T.dot(L,eps)

##This converts the noise signal into the basioc matrix required for Covariance calculation
    def create_mat(self,time,size):
        #noise is numpy type
        diffM = np.zeros(shape=(time,size))
        for i in range(0,size):
            for j in range(0,size):
                diffM[i,j] = i-j
        return np.abs(diffM)

if __name__ == '__main__':
    print "Testing Gausian process sample generation"
    gp = GP(size=20)
    gp_func = theano.function([],gp.sample)
    print gp_func()
