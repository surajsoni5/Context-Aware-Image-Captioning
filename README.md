# CV_Project

## Objective
To produce pragamtic, context aware discriptions of images. We solve the following two problems here.
- ***Justification***:  
    - Given an image, a target (ground-truth) class, and a distractor class, generates a sentence that describes why the target image belongs to the target class rather than distractor class.
- ***Discriminative image captioning***
    - Given two similar images, generates a sentence that describes the target image in context of the semantically similar distractor image. 

## Approach
We training our model using generic **context-agnostic** data (captions that describe a concept or an image in isolation), and use an inference techiqiue called **Emitter-Suppressor Beam Search** to produce context aware image captions. Our models develops upon the architecture of "Show, Attend and Tell". For justification, apart from the image, the model is also conditioned on target-class. 

## Dataset
We use the [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset which contains images of birds and their discriptions. The dataset has 200 classes , each class has 30 images and each image has 10 discriptions. 

## Implementation details
- ***Encoder:*** 
    - We use a pretrained ResNet-101 already available in PyTorch's `torchvision`  module. Discarded the last two layers (pooling and linear layers), since we only need to encode the image, and not classify it.

- ***Decoder:*** 
  - We use a lstm with input embedding of 512 and hidden states of size 1800. For justification the class is embeded into a 512 size vector. 

-  ***Attention*** 
    - We used adaptive pooling over encoder to get an `14*14*512` vector from the encoder, and then apply a linear layer with relu to get the attention weights. We used the soft version of attention. 

-  We use Adam's optimizer, with learning rate of 0.002 which is annealed every 5 epochs. We use dropout with with p = 0.5 . The batch size used was 64, and the number of epochs were 100. GTX 1060.

## Results 
| Image| Target class  | distractor class  |   Caption |
|:---:|---|---|---|
|![](Black_Footed_Albatross_0001_796111.jpg)  |  ![](Capture.PNG) |![](Capture.PNG)   | Fafafafa jkbjbe bhbadhf bjdBHBHJA   | 
|   |   |   |   |   
|   |   |   |   |   


![](https://latex.codecogs.com/png.latex?p=0.5)

<img src="https://latex.codecogs.com/png.latex?p=0.5" />



## Discussion 

## References 
1. Paper: [Context-aware Captions from Context-agnostic Supervision](https://arxiv.org/pdf/1701.02870.pdf)
2. Dataset: [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)


## Requirements and Dependencies

## Setup 

## Files and Folders 
1. Loading data
* [datasets.py](datasets.py) 
* [datapreprocess.py](datapreprocess.py)

2. Model Related
* [models.py](models.py)
* [beamsearch.py](beamsearch.py)

3. Training 
* [train.py](train.py)
* [train_justify.py](train_justify.py)

4. Utils 
* [utils.py](utils.py)

