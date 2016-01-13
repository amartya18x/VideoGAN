import timeit

import numpy

import theano
import theano.tensor as T

from Convolution import ConvPoolLayer
from conv_mlp import HiddenLayer

class Generator(object):
    def __init__(self,x,time_step,learning_rate=0.0000001, n_epochs=200,
                 nkerns=[96, 128, 96 , 96, 128, 96]):

        rng = numpy.random.RandomState(23455)
        
        # allocate symbolic variables for the data
        # start-snippet-1
        x = x   # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels

                            ######################
                            # BUILD ACTUAL MODEL #
                            ######################
        print '... building the model'

        # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # (28, 28) is the size of MNIST images.
        layer0_input = x.reshape((time_step, 1, 60, 80))
        layer0_input.name="layer0_input"
        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
        layer0 = ConvPoolLayer(
            rng,
            input=layer0_input, 
            image_shape=(time_step, 1, 60, 80),
            filter_shape=(nkerns[0], 1, 5, 5),
            poolsize=(2, 2)
        )
        layer0.output.name="layer0_output"
        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (time_step, nkerns[1], 4, 4)
        layer1_input = T.nnet.relu(layer0.output)
        layer1 = ConvPoolLayer(
            rng,
            input=layer1_input,
            image_shape=(time_step, nkerns[0], 28, 38),
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            poolsize=(2, 2)
        )
        layer1.output.name="layer1_output"
        layer2_input = T.nnet.relu(layer1.output)
        layer2 = ConvPoolLayer(
            rng,
            input=layer2_input,
            image_shape=(time_step,nkerns[1],12,17),
            filter_shape=(nkerns[2],nkerns[1],5,5),
            poolsize=(1,1)
        )
        layer2.output.name="layer2_output"
        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (time_step, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (time_step, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        latent_input = T.nnet.relu(layer2.output.flatten(2))
        latent_input.name="latent_input"
        # construct a fully-connected sigmoidal layer
        self.latent = HiddenLayer(
            rng,
            input=latent_input,
            n_in=nkerns[2] * 8 *13 ,
            n_out=500,
            activation=T.tanh
        )

        # classify the values of the fully-connected sigmoidal layer
        layer3 = HiddenLayer(
            rng,
            input=self.latent.output,
            n_in=500,
            n_out=nkerns[3]*11*13,
            activation=T.tanh
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
            poolsize=(1,1)
        )
        layer4.output.name="layer4_output"
        layer5_input = T.nnet.relu(T.repeat(T.repeat(layer4.output,(2),axis=2),2,axis=3))
        layer5_input.name="layer5_inp"
        layer5 = ConvPoolLayer(
            rng,
            input = layer5_input,
            image_shape=(time_step,nkerns[4],36,46),
            filter_shape=(nkerns[5],nkerns[4],5,5),
            poolsize=(1,1)
        )
        layer5.output.name="layer5_output"
        layer6_input = T.nnet.relu(T.repeat(T.repeat(layer5.output,(2),axis=2),2,axis=3))
        layer6_input.name="layer6_input"
        layer6 = ConvPoolLayer(
            rng,
            input = layer6_input,
            image_shape = (time_step,nkerns[5],64,84),
            filter_shape=(1,nkerns[5],5,5),
            poolsize=(1,1)
        )
        self.layer6_output = T.nnet.sigmoid(layer6.output.reshape((time_step,1,60,80)))
        self.layer6_output.name="layer6_output"
        #self.out_var=[self.layer6.output,layer0_input]
        # the cost we minimize during training is the NLL of the model
        self.cost = T.sum((layer6.output-layer0_input)**2)
        
        self.params = layer6.params + layer5.params + layer4.params+ layer3.params + layer2.params + layer1.params + layer0.params + self.latent.params
        self.encoder_params = layer0.params + layer1.params + layer2.params + self.latent.params
        
        # create a list of gradients for all model parameters
        grads = T.grad(self.cost, self.params)
        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        self.update = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(self.params, grads)
        ]
        print "Total Generator Completed"
