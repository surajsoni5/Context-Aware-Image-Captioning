# Context-aware Captions from Context-agnostic Supervision

## Objective
Produce pragmatic, context aware descriptions of images (captions  that  describe differences between images or visual concepts) using context agnositic data (captions that describe  a  concept  or  an  image  in  isolation). We attempt the following two problems.
- ***Justification***:  
    - Given an image, a target (ground-truth) class, and a distractor class, describe the target image to explain why it belongs to the target class, and not the distractor class.
- ***Discriminative image captioning***
    -  Given two similar images, produce a sentence to identify a target image from the distractor image.

## Approach
We trained our model using generic **context-agnostic**  data (captions that describe a concept or an image in isolation), in an encoder-decoder paradigm along with attention, and used an inference techiqiue called **Emitter-Suppressor  Beam Search** to produce context aware image captions. Our model develops upon the architecture of [Show attend and tell](https://arxiv.org/pdf/1502.03044.pdf). For justification, apart from the image, the decoder is also conditioned on target-class. 
#### Emitter-Suppressor Beam Search Algorithm
<p align="center">
    <img src="result_images/es.png" alt="Image" width="400" height="300" />

</p>

## Dataset
We have used the [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset which contains images of birds and their descriptions. The dataset has 200 bird classes (species), each class has 30 images and each image has 10 descriptions. The descriptions are mostly about the morphology of the birds i.e., details about various parts of their body.

<p align="center">
    <img src="result_images/birdlables.png" alt="Image" width="400" height="300" />
</p>
 
## Implementation details 
- ***Encoder*** 
    - We used a pretrained `ResNet-34` already available in the PyTorch's `torchvision`  module and discarded the last two layers (pooling and linear layers), since we only need to encode the image, and not classify it.

- ***Decoder*** 
  - We used LSTM's with `input embedding of size 512` and `hidden states of size 1800`. For justification the class is embeded into a `512 size vector`.   

-  ***Attention*** 
    - We used adaptive pooling over encoder to get a `14*14*512` vector from the encoder and then applied a linear layer with ReLu activation to get the attention weights. Note that we used the soft version of the attention. 

-  We used Adam's optimizer with `learning rate of 0.002` which is annealed every 5 epochs. We used dropout with `p = 0.5`. The `batch size used was 64` and the `number of epochs were 100`. The model was trained on GTX 1060 for 15 hours.

## Results 

### Context Agnostic Captioning 
| Image| Context Agnostic Caption |
|:---:|:---:|
| <img src="result_images/images_caption/Pied_Kingfisher_0002_71698.jpg" width=200 height=175 align="bottom">   | this bird has a white belly and breast with black superciliary and crown |
| <img src="result_images/images_caption/Laysan_Albatross_0049_918.jpg" width=200 height=175>   | this bird has a white head and breast with grey wings and a yellow beak |
| <img src="result_images/images_caption/Common_Yellowthroat_0015_190556.jpg" width=200 height=175>   | this bird has a yellow belly and breast with a black superciliary and gray crown |
| <img src="result_images/images_caption/Red_Legged_Kittiwake_0071_73800.jpg" width=200 height=175>   | this bird has a white crown a white breast and grey wings with black edges |
| <img src="result_images/images_caption/Purple_Finch_0117_27427.jpg" width=200 height=175>   | this bird has a red crown a short bill and a red breast |
| <img src="result_images/images_caption/Pelagic_Cormorant_0012_23565.jpg" width=200 height=175>   | this bird has a long neck and a long bill |
| <img src="result_images/images_caption/Pied_Kingfisher_0004_72135.jpg" width=200 height=175>   | this bird has a long black bill and a black and white spotted body |
| <img src="result_images/images_caption/Long_Tailed_Jaeger_0005_797062.jpg" width=200 height=175>   | this bird has a black crown a white breast and a large wingspan |
| <img src="result_images/images_caption/Western_Grebe_0007_36074.jpg" width=200 height=175>   | this bird has a long yellow bill a black crown and red eyes |
| <img src="result_images/images_caption/Brewer_Sparrow_0008_796703.jpg" width=200 height=175>   | this bird has a white belly and breast with a brown crown and white wingbars |

### Justification Captioning 

| Image| Target class  | distractor class  |   Caption |
|:---:|:---:|:---:|:---:|
|<img src="result_images/images_justify/images/052_Pied_Billed_Grebe_0038_35798.jpg" width=225 height=200>   |  <img src="result_images/images_justify/classes/52.jpg" width=300 height=250>  |<img src="result_images/images_justify/classes/53.jpg" width=300 height=250>    | this bird has a brown crown brown primaries and a brown throat | 
|<img src="result_images/images_justify/images/193_Bewick_Wren_0060_185366.jpg" width=225 height=200>      | <img src="result_images/images_justify/classes/193.jpg" width=300 height=250>  |<img src="result_images/images_justify/classes/199.jpg" width=300 height=250>    | this bird has a white belly and breast with a brown crown and wing | 
|<img src="result_images/images_justify/images/063_Ivory_Gull_0050_49245.jpg" width=225 height=200>        |  <img src="result_images/images_justify/classes/63.jpg" width=300 height=250>  |<img src="result_images/images_justify/classes/60.jpg" width=300 height=250>    | this bird has a white crown as well as a black bill | 
|<img src="result_images/images_justify/images/120_Fox_Sparrow_0135_115251.jpg" width=225 height=200>   |  <img src="result_images/images_justify/classes/120.jpg" width=300 height=250>  |<img src="result_images/images_justify/classes/126.jpg" width=300 height=250>    | this bird has a brown crown brown primaries and a brown belly | 
|<img src="result_images/images_justify/images/075_Green_Jay_0028_65719.jpg" width=225 height=200>   |  <img src="result_images/images_justify/classes/75.jpg" width=300 height=250>  |<img src="result_images/images_justify/classes/74.jpg" width=300 height=250>    | this bird has a blue crown green primaries and a yellow belly | 
|<img src="result_images/images_justify/images/170_Mourning_Warbler_0042_166493.jpg" width=225 height=200>   |   <img src="result_images/images_justify/classes/170.jpg" width=300 height=250>  |<img src="result_images/images_justify/classes/171.jpg" width=300 height=250>    | this bird has a yellow belly and breast with a black neck and crown | 
|<img src="result_images/images_justify/images/056_Pine_Grosbeak_0002_38214.jpg" width=225 height=200>   |   <img src="result_images/images_justify/classes/56.jpg" width=300 height=250>  |<img src="result_images/images_justify/classes/57.jpg" width=300 height=250>    | this bird has a red crown red primaries and a red belly | 
|<img src="result_images/images_justify/images/124_Le_Conte_Sparrow_0024_795190.jpg" width=225 height=200>   |   <img src="result_images/images_justify/classes/124.jpg" width=300 height=250>  |<img src="result_images/images_justify/classes/118.jpg" width=300 height=250>    | this bird has a pointed yellow bill with a yellow breast | 
|<img src="result_images/images_justify/images/182_Yellow_Warbler_0016_176452.jpg" width=225 height=200>   |   <img src="result_images/images_justify/classes/182.jpg" width=300 height=250>  |<img src="result_images/images_justify/classes/159.jpg" width=300 height=250>    | this bird has a yellow crown a short bill and a yellow breast | 
|<img src="result_images/images_justify/images/144_Common_Tern_0009_149609.jpg" width=225 height=200>   |   <img src="result_images/images_justify/classes/144.jpg" width=300 height=250>  |<img src="result_images/images_justify/classes/142.jpg" width=300 height=250>    | this bird has a black crown white primaries and a white belly | 

### Discriminative Captioning 
| Target Image | Distractor Image | Caption | 
|:--:|:--:|:--:|
| <img src="result_images/images_desc/image_target/Black_Footed_Albatross_0032_796115.jpg" width=175 height=175>  | <img src="result_images/images_desc/image_distractor/Laysan_Albatross_0047_619.jpg" width=175 height=175>  | this bird is brown in color over all of its body except for its wings and tail that have white around them | 
| <img src="result_images/images_desc/image_target/Sooty_Albatross_0054_796347.jpg" width=175 height=175>  | <img src="result_images/images_desc/image_distractor/Laysan_Albatross_0040_472.jpg" width=175 height=175>  | this bird has wings that are gray and has a black tail and a black bill | 
| <img src="result_images/images_desc/image_target/Black_Footed_Albatross_0038_212.jpg" width=175 height=175>   | <img src="result_images/images_desc/image_distractor/Laysan_Albatross_0058_637.jpg" width=175 height=175>  | this bird has a brown crown brown primaries and a brown throat | 
| <img src="result_images/images_desc/image_target/Crested_Auklet_0010_794907.jpg" width=175 height=175>   | <img src="result_images/images_desc/image_distractor/Least_Auklet_0014_1901.jpg" width=175 height=175>  | this bird has webbed feet with a bright orange wide beak and jet black over the rest of its body | 
| <img src="result_images/images_desc/image_target/Red_Winged_Blackbird_0024_4180.jpg" width=175 height=175>  | <img src="result_images/images_desc/image_distractor/Brewer_Blackbird_0064_2290.jpg" width=175 height=175>  | this bird has blackhead and body but there is fins of feathers off of the wing that are black white and red | 
| <img src="result_images/images_desc/image_target/Lazuli_Bunting_0030_14986.jpg" width=175 height=175>  | <img src="result_images/images_desc/image_distractor/Indigo_Bunting_0036_13716.jpg" width=175 height=175>  | this bird has a white belly brown breast blue head and white wingbars | 
| <img src="result_images/images_desc/image_target/Black_Billed_Cuckoo_0037_795330.jpg" width=175 height=175>  | <img src="result_images/images_desc/image_distractor/Yellow_Billed_Cuckoo_0018_26535.jpg" width=175 height=175>  | a small brown bird with a white throat and red eyerings | 
| <img src="result_images/images_desc/image_target/Yellow_Bellied_Flycatcher_0017_795490.jpg" width=175 height=175>  | <img src="result_images/images_desc/image_distractor/Scissor_Tailed_Flycatcher_0013_42024.jpg" width=175 height=175>  | a small green bird with a yellow breast and yellow bill | 
| <img src="result_images/images_desc/image_target/Blue_Jay_0080_61617.jpg" width=175 height=175> |  <img src="result_images/images_desc/image_distractor/Florida_Jay_0009_64723.jpg" width=175 height=175>  | this bird has a white belly and breast with blue wings a light gray eyebrow on the head of a front of black and white striped on the wings and bright pale blue rectrices | 

<!-- ![](https://latex.codecogs.com/png.latex?p=0.5)

<img src="https://latex.codecogs.com/png.latex?p=0.5" /> -->


<!-- ## Discussion 
It can be seen that context aware captions gives more information about the image than context agnositic captions.  -->

## Requirements

Kindly use the requirements.txt to set up your machine for replicating this project, some dependencies are :
```
h5py==2.9.0   
matplotlib==3.0.3   
nltk==3.4.1     
numpy==1.16.2  
pandas==0.24.2  
pillow==5.3.0     
python==3.7.3   
pytorch==1.0.0   
torchfile==0.1.0   
torchvision==0.2.1   
tqdm==4.31.1  
```
You can install these dependencies using `pip install -r requirements.txt`

## Setup 
#### Training
```
python datapreprocess.py \path\to\data\set \path\to\vocab\
python train.py
python train_justify.py 
```
#### Testing
Download the pretrained models [checkpoint_d](https://drive.google.com/open?id=1w4zF82hgbPmU9hAHsY92myjOU6HYD_BI) and [checkpoint_j](https://drive.google.com/open?id=1QLyqU5HZHYAyRJTSsIBPqG8zARU0oZUL)
1) Context agnostic captioning: 
` python beamsearch.py c image_path`
2) Justification:
` python beamsearch.py cj target_image_path target_class_path distractor_class_path `
3) Discrimination:
` python beamsearch.py cd target_image_path distractor_image_path `

<!-- ## Files and Folders  -->

## References 
1. Paper: [Context-aware Captions from Context-agnostic Supervision](https://arxiv.org/pdf/1701.02870.pdf)
2. Dataset:
    - Images   : [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
    - Captions :  [Reed et al.](https://arxiv.org/abs/1605.05395) 
3. A beautiful tutorial on [Show, Attend and Tell Implementation](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
