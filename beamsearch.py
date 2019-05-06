import torch
import torchvision.transforms as transforms
from models import *
import torchfile as tf
from scipy.misc import imread, imresize
from PIL import Image
import torch.nn.functional as F
import sys

start=0
end=1

def beam_search(encoder, decoder, image_path, beam_size):
    k = beam_size
    vocab_size = 5725+1 ###

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[start]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    # complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)


        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != end]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    # alphas = complete_seqs_alpha[i]
    return seq

def beam_search_justify_main(encoder, decoder, image_path ,class_t ,class_d,lambda_, beam_size):

    k = beam_size
    vocab_size = 5725+1 ###

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)
    class_embedding_t=decoder.class_embedding(torch.LongTensor([[class_t]]).to(device))
    class_embedding_d=decoder.class_embedding(torch.LongTensor([[class_d]]).to(device))
    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
    class_embedding_t=class_embedding_t.expand(k,1,512)
    class_embedding_d=class_embedding_d.expand(k,1,512)
    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[start]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    # complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h_t, c_t = decoder.init_hidden_state(encoder_out,class_embedding_t)
    h_d, c_d = decoder.init_hidden_state(encoder_out,class_embedding_d)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe_t = decoder.attention(encoder_out, h_t)  # (s, encoder_dim), (s, num_pixels)
        awe_d = decoder.attention(encoder_out, h_d)  # (s, encoder_dim), (s, num_pixels)


        gate_t = decoder.sigmoid(decoder.f_beta(h_t))  # gating scalar, (s, encoder_dim)
        awe_t = gate_t * awe_t
        gate_d = decoder.sigmoid(decoder.f_beta(h_d))  # gating scalar, (s, encoder_dim)
        awe_d = gate_d * awe_d
        h_t, c_t = decoder.decode_step(torch.cat([embeddings, awe_t,class_embedding_t[:,0,:]], dim=1), (h_t, c_t))  # (s, decoder_dim)
        h_d, c_d = decoder.decode_step(torch.cat([embeddings, awe_d,class_embedding_d[:,0,:]], dim=1), (h_d, c_d))  # (s, decoder_dim)

        scores_t = decoder.fc(h_t) # (s, vocab_size)
        scores_d = decoder.fc(h_d)
        scores_t  = F.log_softmax(scores_t, dim=1)
        scores_d  = F.log_softmax(scores_d, dim=1)
        scores = scores_t-(1-lambda_)*scores_d

        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != end]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        h_t = h_t[prev_word_inds[incomplete_inds]]
        c_t = c_t[prev_word_inds[incomplete_inds]]

        h_d = h_d[prev_word_inds[incomplete_inds]]
        c_d = c_d[prev_word_inds[incomplete_inds]]

        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]

        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        class_embedding_t=class_embedding_t[incomplete_inds,:,:]
        class_embedding_d=class_embedding_d[incomplete_inds,:,:]

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    # alphas = complete_seqs_alpha[i]
    return seq



def beam_search_discriminative(encoder, decoder, image_path_t,image_path_d,lambda_, beam_size=3):

    k = beam_size
    vocab_size = 5725+1 ###

    # Read image and process
    img = imread(image_path_t)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out_t = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)

    img = imread(image_path_d)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    image = transform(img)  # (3, 256, 256)
    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out_d = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)


    enc_image_size = encoder_out_t.size(1)
    encoder_dim = encoder_out_t.size(3)

    # Flatten encoding
    encoder_out_t = encoder_out_t.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    encoder_out_d = encoder_out_d.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out_t.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out_t = encoder_out_t.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
    encoder_out_d = encoder_out_d.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[start]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)


    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h_t, c_t = decoder.init_hidden_state(encoder_out_t)
    h_d, c_d = decoder.init_hidden_state(encoder_out_d)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe_t= decoder.attention(encoder_out_t, h_t)  # (s, encoder_dim), (s, num_pixels)
        awe_d= decoder.attention(encoder_out_d, h_d)  # (s, encoder_dim), (s, num_pixels)


        gate_t = decoder.sigmoid(decoder.f_beta(h_t))  # gating scalar, (s, encoder_dim)
        awe_t = gate_t * awe_t
        gate_d = decoder.sigmoid(decoder.f_beta(h_d))  # gating scalar, (s, encoder_dim)
        awe_d = gate_d * awe_d
        
        h_t, c_t = decoder.decode_step(torch.cat([embeddings, awe_t], dim=1), (h_t, c_t))  # (s, decoder_dim)
        h_d, c_d = decoder.decode_step(torch.cat([embeddings, awe_d], dim=1), (h_d, c_d))  # (s, decoder_dim)

        # scores = decoder.fc(h_t)- (1-lambda_)*decoder.fc(h_d)  # (s, vocab_size)
        scores = F.log_softmax(decoder.fc(h_t), dim=1) -(1-lambda_)*F.log_softmax(decoder.fc(h_d), dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != end]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            # complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]

        h_t = h_t[prev_word_inds[incomplete_inds]]
        c_t = c_t[prev_word_inds[incomplete_inds]]
        encoder_out_t = encoder_out_t[prev_word_inds[incomplete_inds]]

        h_d = h_d[prev_word_inds[incomplete_inds]]
        c_d = c_d[prev_word_inds[incomplete_inds]]
        encoder_out_d = encoder_out_d[prev_word_inds[incomplete_inds]]


        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    return seq


if __name__ == "__main__":
    if sys.argv[1]=="c": ## Caption image_path 
        checkpoints=torch.load('checkpoint_d')
        encoder=checkpoints['encoder']
        decoder=checkpoints['decoder']
        encoder.eval()
        decoder.eval()
        image_path=sys.argv[2]
        word_map=tf.load('C:/Users/hello/Desktop/Accads/cvpr2016_cub/vocab_c10.t7',force_8bytes_long=True)
        word_map={word_map[i]:i for i in word_map}
        seq=beam_search(encoder,decoder,image_path,1)
        for i in seq[1:]:
            print(word_map[i].decode("utf-8") ,end=" ")
        print("")
    elif sys.argv[1]=="cj": ## cj Image_path target_class distractor class
        checkpoints=torch.load('checkpoint_j')
        encoder=checkpoints['encoder']
        decoder=checkpoints['decoder'] 
        encoder.eval()
        decoder.eval()
        image_path=sys.argv[2]
        word_map=tf.load('C:/Users/hello/Desktop/Accads/cvpr2016_cub/vocab_c10.t7',force_8bytes_long=True)
        word_map={word_map[i]:i for i in word_map}
        seq=beam_search_justify_main(encoder,decoder,image_path,int(sys.argv[3]),int(sys.argv[4]),0.5,1)
        for i in seq[1:]:
            print(word_map[i].decode("utf-8") ,end=" ")
        print("")
    elif sys.argv[1]=="cd": ## cd Image_path_t Image_path_d
        checkpoints=torch.load('checkpoint_j')
        encoder=checkpoints['encoder']
        decoder=checkpoints['decoder']
        encoder.eval()
        decoder.eval()
        image_path_t=sys.argv[2]
        image_path_d=sys.argv[3]
        word_map=tf.load('C:/Users/hello/Desktop/Accads/cvpr2016_cub/vocab_c10.t7',force_8bytes_long=True)
        word_map={word_map[i]:i for i in word_map}
        seq=beam_search_discriminative(encoder,decoder,image_path_t,image_path_d,0.5,3)
        for i in seq[1:]:
            print(word_map[i].decode("utf-8") ,end=" ")
        print("")