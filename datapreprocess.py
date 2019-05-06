"""
Converts the training data into hdf5 file format, for efficiency during training.
"""
import h5py
import os
import numpy as np
import torchfile as tf
from PIL import Image
from matplotlib.pyplot import imshow
import sys    
cub_dataset    = sys.argv[1] #'C:/Users/hello/Desktop/Accads/CUB_200_2011/' # Location of cub dataset 
captions_dataset= sys.argv[2] # 'C:/Users/hello/Desktop/Accads/cvpr2016_cub/word_c10/' # Location of captions dataset
VocabSize=5725+1 # Starts from 1 and ends at 5725, include start token also
start=0 # 0 is start token
end=1 # 1 is end token
with h5py.File(os.path.join('dataset.hdf5'), 'w') as h:
    h['capsperimage']=10
    maxcapslen=30 # Max size of captions
    images = h.create_dataset('images', (11788, 3, 256, 256), dtype='uint8') 
    captions = h.create_dataset('captions', (11788 ,10,maxcapslen+2), dtype='int32')
    class_ =  h.create_dataset('class', (11788,), dtype='int32')
    prev=''
    last=0
    captions[:]=1
    with open(cub_dataset + 'images.txt') as f:
        for i,line in enumerate(f):
            line = line.strip('\n').split(' ')[1]
            img = Image.open(cub_dataset+'images/'+line)
            img = img.resize((256,256))
            img = np.asarray( img, dtype='int32' )
            # The resnet accepts 3*w*w
            if img.shape!=(256,256,3):  # A few of the images are black and white
                img=np.array([img,img,img])
                print(line)
            else:
                img = img.transpose(2, 0, 1) 
            images[i]=img
            present=line.split('/')[0]
            if prev!=present:
                prev=present
                temp=tf.load(captions_dataset+line.split('/')[0]+'.t7',force_8bytes_long=True)
                temp=temp.transpose(0,2,1)
                captions[last:last+temp.shape[0],:,0]=start 
                captions[last:last+temp.shape[0],:,1:maxcapslen+1]=temp
                last =last+temp.shape[0]
            class_[i]=int(line[:3])
    h['numcaptions']=i*10 
