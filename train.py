import torch
from datasets import CaptionDataset
import torchvision.transforms as transforms
import torch.optim as optim
from models import *
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

batch_size=64 # Batch size
workers=0 # Workers for loading the dataset. Need this to be 0 for windows, change to sutable value for other os.
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 1800  # dimension of decoder RNN
dropout = 0.5 # Dropout rate
decoder_lr = 2*1e-3 # Decoder learning rate
numepochs=100 # Number of epochs
load=False ## Make this false when you don't want load a checkpoint 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(\
        CaptionDataset( transform=transforms.Compose([normalize])),\
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True) 
        # Note that the resize is already done in the encoder, so no need to do it here again
if load:
	# Load the model from checkpoints 
	checkpoints=torch.load('checkpoint_d')
	encoder=checkpoints['encoder']
	decoder=checkpoints['decoder']
	decoder_optimizer=checkpoints['decoder_optimizer']
	epoch=checkpoints['epoch']
	decoder_lr=decoder_lr*pow(0.8,epoch//5)
	for param_group in decoder_optimizer.param_groups:
		param_group['lr'] = decoder_lr
else:
	epoch=0
	encoder=Encoder()
	decoder=DecoderWithAttention(attention_dim=attention_dim,embed_dim=emb_dim,decoder_dim=decoder_dim,vocab_size=5725+1)
	decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),lr=decoder_lr)

criterion = nn.CrossEntropyLoss()
encoder=encoder.to(device)
decoder=decoder.to(device)
criterion=criterion.to(device)

decoder.train()
encoder.train()
for epoch in range(epoch,numepochs):
	if epoch%5==0 and epoch>0:
		# For every 5 epochs, the lr is annealed by 0.8
		decoder_lr*=0.8
		for param_group in decoder_optimizer.param_groups:
			param_group['lr'] = decoder_lr
	for i,(img,caption,caplen,class_k) in tqdm(enumerate(train_loader),desc="Batch"):

		img=img.to(device)
		caption=caption.to(device)
		caplen=caplen.to(device)
		class_k=class_k.to(device)

		img = encoder(img)
		scores, caps_sorted, decode_lengths, sort_ind = decoder(img, caption, caplen)
		targets = caps_sorted[:, 1:]
		# Suitable format, so that loss can be applied. The scores had unwated padding, that is removed. Similarly for target
		scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True) #[32, 30, 5726] to [960, 5726]
		targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True) #[32, 30] to 9[60]

		# A gradient decent step
		loss=criterion(scores,targets) 
		decoder_optimizer.zero_grad()
		loss.backward()
		decoder_optimizer.step()

		tqdm.write(f"Loss {loss.detach().cpu().numpy()}")
		
	### Save model.  Checkpoints ####
	state = {
			'epoch': epoch,
			'encoder': encoder,
			'decoder': decoder,
			'decoder_optimizer': decoder_optimizer
			}
	filename = 'checkpoint_d'
	torch.save(state, filename)
	##################################
