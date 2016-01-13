import cv2
import numpy
import theano
import theano.tensor as T
from GP_MLP import GP_MLP
from Autoencoder import Autoencoder
from read import VideoReader as VR
from Generator import Generator
from theano.tensor.shared_randomstreams import RandomStreams
theano.config.optimizer='None'
theano.config.floatX='float32'
def cross_entropy(z,x):
    y = T.sum((z-x)**2,axis=1)
    return T.mean(y)
class Discriminator(object):
    def __init__(self,memory,time_step,h_size,n_hid,mlp_hid,lr,video_inp,video_size,srng):
        self.gp_mlp = GP_MLP(memory = memory,
                             srng=srng,
                        n_in = time_step,
                        n_hid = mlp_hid,
                             n_out = h_size)
        #self.gen = Autoencoder(video_size,h_size,video_inp)
        self.gen= Generator(x=video_inp,time_step = time_step)
        self.n_in = h_size
        self.n_hid = n_hid
        self.n_out = 1
        self.srng = srng
        self.factor = 10
        u_gp = self.gp_mlp.mlp_out
        u_gp.name = "GP_input"
        u_auto = self.gen.latent.output
        u_auto.name="Auto_latent"
        #u_auto = self.gen.latent
        self.get_auto = theano.function([video_inp],u_auto)
        # initial hidden state of the RNN
        h0 = theano.shared(numpy.zeros(self.n_hid,),name="RNN_h0")
        # learning rate
        lr = lr
        # recurrent weights as a shared variable
        W = theano.shared(numpy.random.normal(0,0.01,size=(self.n_hid, self.n_hid) ),name="W_hh")
        b = theano.shared(numpy.zeros(shape=(self.n_hid,)),name="b_hh")
        # input to hidden layer weights
        W_in = theano.shared(numpy.random.normal(0,0.01,size=(self.n_in, self.n_hid) ),name="W_in")
        b_in = theano.shared(numpy.zeros(shape=(self.n_hid,)),name="b_in")
        # hidden to output layer weights
        W_out = theano.shared(numpy.random.normal(0,0.01,size=(self.n_hid, self.n_out) ),name="W_out")
        b_out = theano.shared(numpy.zeros(shape=(self.n_out,)),name="b_out")
        pos_time = theano.shared(numpy.ones((time_step,1)),name="pos")
        neg_time = theano.shared(numpy.zeros((time_step,1)),name="neg")
        [_, y_gp], _ = theano.scan(self.step,
                        sequences=u_gp,
                        outputs_info=[h0, None],
                        non_sequences=[W, W_in, W_out,b_in,b_out])
        [_, y_auto], _ = theano.scan(self.step,
                        sequences=u_auto,
                        outputs_info=[h0, None],
                        non_sequences=[W, W_in, W_out,b_in,b_out])
        # the hidden state `h` for the entire sequence, and the output for the
        # entrie sequence `y` (first dimension is always time)
        # error between output and target
        #error_lat = ((y_latent - pos_time) ** 2).sum()
        y_gp.name="output_gp_RNN"
        #error_pos = (cross_entropy(y_gp, pos_time))*1.0
        #cross_entr = (cross_entropy(y_gp, neg_time)  + cross_entropy(y_auto, pos_time))/2
        error_pos = T.sum(T.sum(((y_gp-pos_time))**2))*1.0
        error_posR = error_pos + 0.0001*((self.gp_mlp.mlp.W0**2).sum()+(self.gp_mlp.mlp.W1**2).sum()+(self.gp_mlp.mlp.W2**2).sum()+(self.gp_mlp.mlp.b0**2).sum()+(self.gp_mlp.mlp.b1**2).sum()+(self.gp_mlp.mlp.b2**2).sum())
        cross_entr = (T.sum((y_gp- neg_time)**2)  + T.sum((y_auto- pos_time)**2))/2
        error_neg = cross_entr + 0.001*(abs(W**2).sum())+(abs(W_in**2).sum())+((W_out**2).sum())+((b_in**2).sum())+((b_out**2).sum())
        # gradients on the weights using BPTT
        gW, gW_in, gW_out = T.grad(error_neg, [W, W_in, W_out])
        # training function, that computes the error and updates the weights using
        # SGD.
        grads=[]
        params=[W,W_in,W_out,b_in,b_out]
        print [x.dtype for x in params]
        for param in params:
            grads.append(T.grad(error_neg,param))
        print [x.dtype for x in grads]
        print error_neg.dtype
        self.update = []
        for param,grad in zip(params,grads):
            self.update.append((param,param-grad*lr))
        

        gp_grads=[]
        for param in self.gp_mlp.params:
            gp_grads.append(T.grad(error_posR,param))
        self.gp_update = []
        for param,grad in zip(self.gp_mlp.params,gp_grads):
            self.gp_update.append((param,param-grad*lr*self.factor))

        self.disc_auto_update=[]
        disc_auto_grads=[]
        for param in self.gen.encoder_params:
            disc_auto_grads.append(T.grad(error_neg,param))
        for param,grad in zip(self.gen.encoder_params,disc_auto_grads):
            self.disc_auto_update.append((param,param-grad*lr))

            
        self.train_auto = theano.function([video_inp],self.gen.cost,updates=self.gen.update,allow_input_downcast=True)
        self.train_disc_auto = theano.function([video_inp],error_neg,updates=self.disc_auto_update,allow_input_downcast=True)
        self.train_disc = theano.function([video_inp],
                                          error_neg,
                                          updates = self.update,
                                          allow_input_downcast=True)

        self.train_gp = theano.function([],
                                        error_pos,
                                        updates = self.gp_update,allow_input_downcast=True
        )
        
        self.train_out = theano.function([],
                                         y_gp
        )
        self.get_auto = theano.function([video_inp],self.gen.cost,allow_input_downcast=True)
        self.get_disc_auto = theano.function([video_inp],error_neg,allow_input_downcast=True)
        self.get_disc = theano.function([video_inp],
                                          cross_entr,
                                          allow_input_downcast=True)

        self.get_gp = theano.function([],
                                        error_pos,allow_input_downcast=True
        )

        self.total_params = [W,W_in,W_out]+self.gp_mlp.params+self.gen.params+[b_in,b,b_out]
        print "Discrminator built"
    def step(self,u_t, h_tm1, W, W_in, W_out,b_in,b_out):
        h_t1 = T.nnet.relu(T.dot(u_t, W_in)+b_in,alpha=0.2)
        h_t = T.nnet.sigmoid(h_t1+ T.dot(h_tm1, W))
        y_t = T.nnet.sigmoid(T.dot(h_t, W_out)+b_out)
        return h_t, y_t
    
      
if __name__=='__main__':
    vr = VR('../data/person01_boxing_d4_uncomp.avi')
    
    video_inp = T.dmatrix("video_inp")
    input=[]
    for i in range(0,20):
        input.append(numpy.asarray(cv2.resize(vr.next_frame(),(80,60)).flatten()))
    print numpy.min(input),numpy.max(input)
    new_matrix = (input - numpy.min(input))/(numpy.max(input)-numpy.min(input)*1.0)
    input = new_matrix.T
    print input
    disc = Discriminator(memory=1000,time_step=20,h_size=500,n_hid=1000,mlp_hid=500,lr=0.001,video_size=80*60,video_inp=video_inp)
    print disc.get_auto(input).shape
    print disc.train_auto(input)
    print disc.train_disc(input)
    print disc.train_gp()
    for i in range(1,1000):
        print disc.train_auto(input)
        print disc.train_disc(input)
        print disc.train_gp()
    print disc.train_auto(input)
    print disc.train_disc(input)
    print disc.train_gp()
