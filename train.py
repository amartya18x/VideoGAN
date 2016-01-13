import cv2
import numpy
import theano
import theano.tensor as T
from read import VideoReader as VR
from Discriminator import Discriminator
from os import listdir
from os.path import isfile, join
import cPickle
from theano.tensor.shared_randomstreams import RandomStreams
mypath='../data'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
theano.config.optimizer='None'
time_step = 20
eps=0.0
video_inp = T.matrix("video_inp")
srng = RandomStreams(seed=12345)
disc = Discriminator(memory=1000,time_step=time_step,h_size=500,n_hid=1000,mlp_hid=1000,lr=0.0002,video_size=80*60,video_inp=video_inp,srng=srng)
count_vid = 0
disc_err = 0
fool_err =0
patience = 10
benevolent = 2
punishement = 1.0001
threshold = 2.0
flag = 1
for l in range(0,10000):
    for f in onlyfiles:
        vr = VR('../data/'+f)
        count = vr.count_frame()
        vr = VR('../data/'+f)
        for j in range(0,count-time_step-1):
            input=[]
            for i in range(0,time_step):
                input.append(numpy.asarray(cv2.resize(vr.return_frame(j+1),(80,60)).flatten()))
            new_matrix = (input - numpy.min(input))/(numpy.max(input)-numpy.min(input)*1.0)
            input = new_matrix.T
            disc_err = disc.get_disc(input)
            fool_err = disc.get_gp()
            #print disc.train_out()
            if flag ==1 and disc_err <threshold:
                flag = 0
            if flag == 0 and fool_err <threshold:
                flag = 1
                threshold *= 0.9
                
            if flag == 0:
                if patience == 0:
                    disc.factor *= 2
                    patience = 20
                else:
                    patience -= 1
                for i in range(1,benevolent):
                    fooling_err = disc.train_gp()
                reconstruction_err =  disc.train_auto(input)
                if count_vid%10 == 0:
                    print "Epoch no.: [%d] Iteration number: [%d] \n Reconstruction Error :%lf \n \t Discriminator is stopped: %lf, \n \tFooling Error: %lf, \n \t change by training in fooling:%lf" %(l,count_vid,reconstruction_err,disc_err,fool_err,(fooling_err-fool_err))
            if flag == 1:
                discriminator_err = disc.train_disc(input)
                disc.factor = 5
                #disc.train_disc_auto(input)
                reconstruction_err =  disc.train_auto(input)
                if count_vid%10 == 0:
                    print "Epoch no.: [%d] Iteration number: [%d] \n \tDiscrimination error :%lf \n \tReconstruction Erros: %d\n \tFooling is stopped:%lf." %(l,count_vid,disc_err,reconstruction_err,fool_err)
             
            if count_vid%10==0:
                file_name = 'params'+'.pkl'
                f = file(file_name, 'wb')
                cPickle.dump(disc.total_params, f, protocol=cPickle.HIGHEST_PROTOCOL)
                f.close()
            count_vid +=1
        l = l+1




#print disc.train_gp()
#for i in range(1,1000):
#    print disc.train_auto(input)
#    print disc.train_disc(input)
#print disc.train_gp()
#print disc.train_auto(input)
#print disc.train_disc(input)
#print disc.train_gp()
