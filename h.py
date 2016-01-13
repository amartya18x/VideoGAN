import theano
import numpy as np
import theano.tensor as T
theano.config.optimizer='None'
theano.config.exception_verbosity='high'
class Autoencoder(object):
    def __init__(self,n_in,n_hid,inp):
        self.n_in = n_in
        self.n_hid = n_hid
        X = inp
        X.name="Auto_input"
        W0 = theano.shared(np.random.normal(size=(n_hid,n_in)),name="AutoW0")
        W1 = theano.shared(np.random.normal(size=(n_in,n_hid)),name="AutoW1")
	W2 = theano.shared(np.random.normal(size=(n_hid,n_hid)),name=AutoW2")
        b0 = theano.shared(np.zeros((n_hid,)),name="Autob0")
        b1 = theano.shared(np.zeros((n_in,)),name="Autob1")
	b2 = theano.shared(np.shared((n_hid,)),name="Autob2")
        self.latent = T.dot(W0,X)
        self.latent1 = T.tanh(self.latent+ b0.dimshuffle(0,'x'))
	self.latent2 = T.tanh(T.dot(W2,self.latent1,b2.dimshuffle(0,'x')))
        self.y =   T.tanh(T.dot(W1,self.latent2) + b1.dimshuffle(0,'x'))
        self.cost = ((self.y-X)**2).sum()
        grads=[]
        params=[W0,W1,W2,b0,b1,b2]
        for param in params:
            grads.append(T.grad(self.cost,param))
        self.update = []
        for param,grad in zip(params,grads):
            self.update.append((param,param-grad*0.01))
        
    

if __name__ == '__main__':
    x = T.matrix('x')
    auto = Autoencoder(10,1,x)
    input = np.random.uniform(size=(10,2))
    print input
    see_res = theano.function([],auto.y,givens={x:input},updates=auto.update)
    print see_res()
    for i in range(1,50000):
        see_res()
    print see_res()
