"""
Constructs the dataset from hdf5 file, which was created using datapreprocess script
"""
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
  
class CaptionDataset(Dataset):
    def __init__(self, transform):
        self.h = h5py.File('dataset.hdf5','r',driver='core')
        self.capsperimage =  self.h['capsperimage'].value
        self.images = np.array(self.h['images'])  
        self.captions= np.array(self.h['captions'])
        self.class_k =np.array(self.h['class'])
        self.maxcapslen=30+3 
        self.numcaptions=self.h['numcaptions'].value 
        self.transform=transform

    def __getitem__(self, i):
        img = torch.FloatTensor(self.images[i // self.capsperimage] / 255.)
        img = self.transform(img)
        caption = torch.LongTensor(self.captions[i//self.capsperimage,i%self.capsperimage])
        caplen = torch.LongTensor([self.maxcapslen- sum(caption==1)])
        class_k = torch.LongTensor([self.class_k[i // self.capsperimage]])
        return img,caption,caplen,class_k

    def __len__(self): 
        return self.numcaptions

if __name__ == "__main__":
    pass