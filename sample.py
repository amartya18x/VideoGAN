import numpy
import theano
import theano.tensor as T
from GP_MLP import GP_MLP
from Convolution import ConvPoolLayer
from conv_mlp import HiddenLayer
import cv2 as cv2
import cPickle
import Image
from theano.tensor.shared_randomstreams import RandomStreams
theano.config.floatX='float32'
class Sampler(object):
    def __init__(self,memory,time_step,h_size,n_hid,mlp_hid,param,srng):
        self.gp_mlp = GP_MLP(memory = memory,
                             n_in = time_step,
                             n_hid = mlp_hid,
                             n_out = h_size,
                             srng=srng
        )
        rng = numpy.random.RandomState(23455)
        nkerns=[96, 128, 96 , 96, 128, 96]
        u_gp = self.gp_mlp.mlp_out
        u_gp.name = "GP_input"
        print u_gp.dtype
        layer3 = HiddenLayer(
            rng,
            input=u_gp,
            n_in=500,
            n_out=nkerns[3]*11*13,
            activation=T.tanh,
            W=param[0],
            b=param[1]
        )
        layer3_output1 = layer3.output
        layer3_output = layer3_output1.reshape((time_step,nkerns[3],11,13))
        layer3_output.name = "layer3_output"
        layer4_input = T.nnet.relu(T.repeat(T.repeat(layer3_output,(2),axis=2),2,axis=3))
        layer4 = ConvPoolLayer(
            rng,
            input=layer4_input,
            image_shape=(time_step,nkerns[3],22,26),
            filter_shape=(nkerns[4],nkerns[3],5,4),
            poolsize=(1,1),
            W=param[2],
            b=param[3]
        )
        layer4.output.name="layer4_output"
        layer5_input = T.nnet.relu(T.repeat(T.repeat(layer4.output,(2),axis=2),2,axis=3))
        layer5_input.name="layer5_inp"
        layer5 = ConvPoolLayer(
            rng,
            input = layer5_input,
            image_shape=(time_step,nkerns[4],36,46),
            filter_shape=(nkerns[5],nkerns[4],5,5),
            poolsize=(1,1),
            W=param[4],
            b=param[5]
        )
        layer5.output.name="layer5_output"
        layer6_input = T.nnet.relu(T.repeat(T.repeat(layer5.output,(2),axis=2),2,axis=3))
        layer6_input.name="layer6_input"
        print param[6]
        print param[7]
        layer6 = ConvPoolLayer(
            rng,
            input = layer6_input,
            image_shape = (time_step,nkerns[5],64,84),
            filter_shape=(1,nkerns[5],5,5),
            poolsize=(1,1),
            W=param[6],
            b=param[7]
        )
        self.layer6_output = T.nnet.sigmoid(layer6.output.reshape((time_step,1,60,80)))
time_step = 100
param_File = 'params.pkl'
f = open(param_File,'r')
params = cPickle.load(f)
f.close()
mW0 = params[3]
mW1 = params[4]
mb0 = params[5]
mb1 = params[6]
mW2 = params[7]
mb2 = params[8]
gl3w = params[15]
gl3b = params[16]
gl4w = params[13]
gl4b = params[14]
gl5w = params[11]
gl5b = params[12]
gl6w = params[9]
gl6b = params[10]
conv_params=[gl3w,gl3b,gl4w,gl4b,gl5w,gl5b,gl6w,gl6b]
print params

srng = RandomStreams(seed=876)
samp = Sampler(memory=1000,time_step=time_step,h_size=500,n_hid=200,mlp_hid=500,param=conv_params,srng=srng)
samp.gp_mlp.mlp.W0=mW0
samp.gp_mlp.mlp.W1=mW1
samp.gp_mlp.mlp.b0=mb0
samp.gp_mlp.mlp.b1=mb1
samp.gp_mlp.mlp.W2=mW2
samp.gp_mlp.mlp.b2=mb2
sample_fn = theano.function([],samp.layer6_output,allow_input_downcast=True)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('test1.avi',fourcc,20,(60,80))
print sample_fn()
a = sample_fn()
i=0
for frame in a:
    pic = frame[0,:,:]
    x = numpy.repeat(pic,3,axis=1)
    print x.shape
    x = (x.reshape((60, 80, 3))*255.0).astype('u1')
    print x.shape
    im = Image.fromarray(x)
    name ="../image/"+str(i)+"pic.jpeg"
    i = i+1
    im.save(name)
    writer.write(x)

