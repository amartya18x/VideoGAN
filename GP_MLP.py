from Gaussian_Process import GP
from MLP import MLP
import theano
import numpy as np
import theano.tensor as T
theano.config.exception_verbosity='high'
class GP_MLP(object):
    def __init__(self,memory,n_in,n_hid,n_out,srng):
        gp = GP(memory,srng=srng,time_step=n_in)
        self.mlp = MLP(memory,n_hid,n_out,gp.sample)
        self.mlp_out = self.mlp.output
        self.params = self.mlp.params
        print "Total Gaussian Process and MLP built"
if __name__=='__main__':
    gp_mlp = GP_MLP(memory=10,n_in=20,n_hid=40,n_out=30)
    func = theano.function([],gp_mlp.mlp_out)
    print func().shape

