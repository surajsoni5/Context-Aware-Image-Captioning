import torch
from datasets import CaptionDataset
import torchvision.transforms as transforms
import torch.optim as optim
from models import *
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
batch_size=64
workers=0
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 1800  # dimension of decoder RNN
dropout = 0.5
encoder_lr = 1e-4 
decoder_lr = 2*1e-3 
fine_tune_encoder = False
numepochs=100
load=True ###########FIXME: TODO:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(\
        CaptionDataset( transform=transforms.Compose([normalize])),\
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True) #FIXME: Set shuffle true
        # Note that the resize is already done in the encoder, so no need to do it here again
if load:
	checkpoints=torch.load('checkpoint1')
	encoder=checkpoints['encoder']
	decoder=checkpoints['decoder']
	encoder_optimizer=checkpoints['encoder_optimizer']
	decoder_optimizer=checkpoints['decoder_optimizer']
	epoch=checkpoints['epoch']
	encoder_lr=encoder_lr*pow(0.8,epoch//5)
	decoder_lr=decoder_lr*pow(0.8,epoch//5)
	for param_group in decoder_optimizer.param_groups:
		param_group['lr'] = decoder_lr
		print(decoder_lr)
else:
	epoch=0
	encoder=Encoder()
	decoder=DecoderWithAttention(attention_dim=attention_dim,embed_dim=emb_dim,decoder_dim=decoder_dim,vocab_size=5725+1)
	encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
												lr=encoder_lr) if fine_tune_encoder else None
	decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                        lr=decoder_lr)

criterion = nn.CrossEntropyLoss()
encoder=encoder.to(device)
decoder=decoder.to(device)
criterion=criterion.to(device)

# Confirm Dataloader iff working correctly
decoder.train()
encoder.train()
for epoch in range(epoch,numepochs):
	print('fafa')
	if epoch%5==0 and epoch>0:
		decoder_lr*=0.8
		encoder_lr*=0.8
		for param_group in decoder_optimizer.param_groups:
			param_group['lr'] = decoder_lr
	for i,(img,caption,caplen,class_) in enumerate(train_loader):
		img=img.to(device)
	#     print('fafa')
		caption=caption.to(device)
		caplen=caplen.to(device)
		class_=class_.to(device)
		img = encoder(img)
		scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(img, caption, caplen)
		targets = caps_sorted[:, 1:]

		scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True) #[32, 30, 5726] to [960, 5726]
		targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True) #[32, 30] to 9[60]
		loss=criterion(scores,targets) # add gating scalar, to this loss
		# Finetune encoder
		print(loss)
		decoder_optimizer.zero_grad()
		if encoder_optimizer is not None:
			encoder_optimizer.zero_grad()
		
		loss.backward()

		#Clip gradinets
		decoder_optimizer.step()
		if encoder_optimizer is not None:
			encoder_optimizer.step()
			
	state = {
			'epoch': epoch,
			'encoder': encoder,
			'decoder': decoder,
			'encoder_optimizer': encoder_optimizer,
			'decoder_optimizer': decoder_optimizer
			}
	filename = 'checkpoint'+'1'
	torch.save(state, filename)
	
# Sets the evaluation mode
decoder.eval()
encoder.eval()