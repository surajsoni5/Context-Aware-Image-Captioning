# CV_Project

## Objective
Produce pragamtic, context aware discripions of images. We solve the following two problems here.
#### Justification:  
Given an image, a target(ground-truth) class, and a distractor class, describe the target image to explain why it belongs to the target class, and not the distractor class.
#### Discriminative image captioning
Given two similar images, produce a sentence to identify a target image from the distractor image

### Approch
We training our model using generic **context-agnostic**  data (captions that describe a concept or an image in isolation), and use an infernce techiqiue called **Emitter-Suppressor  Beam Search** to produce context aware image captions. Our models developns upon the arctecture of Show attend and tell. For justification, apart from the image, the model is also conditioned on target-class. 

### Dataset
We use the CUB_2011 dataset which contains images of birds and their discriptions. The dataset has 200 classes , each class has 30 images and each image has 10 discriptions. 

### Implementation details
***Encoder:*** We use a pretrained ResNet-101 already available in PyTorch's `torchvision`  module. Discarded the last two layers (pooling and linear layers), since we only need to encode the image, and not classify it.

***Decoder:*** We use am lstm with input embedding of 512 and hidden states of size 1800. For justification the class is embeded into a 512 size vector. 

***Atention*** We used adaptive pooling over encoder to get an `14*14*512` vector from the encoder, and then apply a linear layer with relu to get the attention weights. We used the soft version of attention. 

We use Adam's optimizer, with learning rate of 0.002 which is annealed every 5 epochs. We use dropout with with `p=0.5`. The batch size used was 64, and the number of epochs were 100. GTX 1060.