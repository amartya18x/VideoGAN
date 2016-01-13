import numpy as np
import theano
import theano.tensor as T
from theano.tensor import slinalg as sin
from theano.tensor.shared_randomstreams import RandomStreams
dtype = np.float32
class GP(object):
    def __init__(self,size,srng,v0=1000,v1=1,v2=1,r=0.01,time_step=10,alpha=2):
        self.lsize = size
        self.time = time_step
        self.dtype = theano.config.floatX
        self.v0 = theano.shared(np.cast[dtype](v0),name="v0")
        self.v1 = theano.shared(np.cast[dtype](v1),name="v1")
        self.v2 = theano.shared(np.cast[dtype](v2),name="v2")
        self.alpha = theano.shared(np.cast[dtype](alpha),name="alpha")
        self.gamma = theano.shared(np.cast[dtype](r),name="r")
        self.srng = srng
        self.DiffM = theano.shared(self.create_mat(self.time),name="diffM")
        self.params = [self.v0,self.gamma,self.alpha]
        self.sample = self.return_output(self.DiffM)
        print self.sample.dtype
        print "....Gaussian Process Built"
##This takes the difference matrix and samples from the GP and returns a signal
    def return_output(self,Dif):
        #Dif is theano.Tensor.matrix type
        Frac = Dif/self.gamma
        Cov = self.v0*T.pow(Frac,self.alpha)
        L = sin.cholesky(T.exp(-Cov))
        eps = self.srng.normal(avg=0,std=0.001,size=(self.time,self.lsize))
        return T.dot(L,eps)

##This converts the noise signal into the basioc matrix required for Covariance calculation
    def create_mat(self,size):
        #noise is numpy type
        diffM = np.zeros(shape=(size,size),dtype=np.float32)
        for i in range(0,size):
            for j in range(0,size):
                diffM[i,j] = i-j
        return np.abs(diffM)

if __name__ == '__main__':
    print "Testing Gausian process sample generation"
    srng = RandomStreams(seed=243)
    gp = GP(size=20,srng=srng)
    gp_func = theano.function([],gp.sample)
    print gp_func()
