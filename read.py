import cv2 as cv2
import numpy as np
from PIL import Image as im
from os import listdir
from os.path import isfile, join
class VideoReader(object):
    def __init__(self,filename):
        self.filename = filename
        self.vidcap = cv2.VideoCapture(self.filename)
        self.frame_num=0
    def count_frame(self):
        count =0
        flag =1
        local_vidcap = cv2.VideoCapture(self.filename)
        while flag:
            if local_vidcap.isOpened():
                local_frame = local_vidcap.read()
                flag=local_frame[0]
                count+=1
            else:
                break
        return  count
    def next_frame(self,display=0):
        frame = self.vidcap.read()
        self.frame_num += 1
        if display == 1:
            img = im.fromarray(frame[1])
            img.show()
        return frame[1][:,:,1]
    def return_frame(self,num=0,display=0):
        local_count = 0
        local_vidcap = cv2.VideoCapture(self.filename)
        while(local_count<num):
            local_vidcap.read()
            local_count += 1
        local_frame = local_vidcap.read() 
        if display == 1:
            img = im.fromarray(local_frame)
            img.show()
        return local_frame[1][:,:,1]
mypath="../data"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for files in onlyfiles:
    vr = VideoReader(mypath+"/"+files)
    n = vr.count_frame()/100
#image = vr.return_frame(num=1,display=0)
image=[]
#a =vr.return_frame(num=2,display=0)
#print a.shape
print vr.count_frame()
#for i in range(1,100):
#    image.append(np.asarray(vr.next_frame()).flatten())
#print vr.count_frame()
#print np.asarray(image).shape
