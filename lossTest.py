import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loaderDoc2vec import get_loader
from build_vocab import Vocabulary
from losstestModel import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec
from numpy import linalg as LA
from nltk.tokenize import word_tokenize
import rouge
import random
# import matplotlib.pyplot as plt

import pandas as pd

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d2v= Doc2Vec.load("doc2vec_100dim_science.model")

def Doc2vec(doc):
    test_data = word_tokenize(doc.lower())
    return d2v.infer_vector(test_data)

def cosin(v1,v2):
    if(LA.norm(v1)!=0 and LA.norm(v2)!=0):
        return (np.dot(np.array(v1),np.array(v2))/(LA.norm(v1) * LA.norm(v2)))
    else:
        return 1

def prepare_results(metric, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

def get_embed(predicted):
    return emb_model.wv[vocab.idx2word[predicted]]


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Load vocabulary wrapper
    # global emb_model, vocab

    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # with open(args.emb_model, 'rb') as f:
    #     emb_model = pickle.load(f)

    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab,
                                args.dictionary, args.batch_size,
                                shuffle=True, num_workers=args.num_workers)

    # Build the models
    #encoder = EncoderCNN(args.embed_size).to(device)
    # dictionary = pd.read_csv(args.dictionary, header=0,encoding = 'unicode_escape',error_bad_lines=False)
    dictionary =pd.read_csv(args.dictionary, header=0,error_bad_lines=False)
    dictionary = list(dictionary['keys'])

    decoder = DecoderRNN(len(dictionary), args.hidden_size, len(vocab), args.num_layers).to(device)
    decoder.load_state_dict(torch.load(args.model_path, map_location=device))
    decoder.eval()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # params = list(decoder.parameters())# + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    # optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the models
    # print("dictionary", len(dictionary))
    # print("vocab", len(vocab))
    data = pd.read_csv(args.caption_path, header=0,encoding = 'unicode_escape',error_bad_lines=False)

    total_step = len(data_loader)
    # for epoch in range(args.num_epochs):
    count = 0
    total_loss = 0
    rand_loss = 0
    # reference = []
    # hypothesis = []
    for i, (array, captions, lengths, word_text) in enumerate(data_loader):
        # reference.append(word_text[0])
        rand_text = data.iloc[random.randint(0, data.shape[0]-1)]['val']

        # Set mini-batch dataset
        array = array.to(device)
        captions = captions.to(device)
        # print("captions", captions.shape)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        # print("targets", targets.shape)
        # outputs = decoder.sampleWordEmb(array, lengths[0])
        outputs = decoder.sampleDoc2vec(array, lengths[0])
        # print('output', outputs[0])
        # print('lengths', len(lengths))
        # print('targets', targets)
        sentence = ''
        for i in range(len(outputs)):
            sampled_ids = outputs[i].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

            # Convert word_ids to words
            sampled_caption = []
            for word_id in sampled_ids:
                word = vocab.idx2word[word_id]
                if(word != '<start>' and word != '<unk>' and word != '<end>'):
                    sampled_caption.append(word)
                if word == '<end>':
                    break
            sentence = sentence.join(' ')
            sentence = sentence.join(sampled_caption)
        # hypothesis.append(sentence)
        # print(word_text[0])
        doc1 = Doc2vec(word_text[0])
        doc2 = Doc2vec(sentence)
        doc3 = Doc2vec(rand_text)
        # print(cosin(doc1, doc2))


        # total_loss += criterion(outputs[0], targets)
        # target_embed = decoder.embed(targets)
        # print("embed targets", target_embed.shape)
        # cosine = cosine_similarity(outputs[0].detach().numpy(), target_embed.detach().numpy())
        # cos = cosine.diagonal()
        # tmp = np.mean(cos)
        total_loss += cosin(doc1, doc2)
        rand_loss += cosin(doc1, doc3)
        count += 1
        # print(count)
    print("Similarity ", total_loss/count)
    print("Random Similarity", rand_loss/count)



    # evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
    #                        max_n=4,
    #                        limit_length=True,
    #                        length_limit=100,
    #                        length_limit_type='words',
    #                        apply_avg=True,
    #                        apply_best=False,
    #                        alpha=0.5, # Default F1_score
    #                        weight_factor=1.2,
    #                        stemming=True)
    #
    # scores = evaluator.get_scores(hypothesis, reference)
    # for metric, results in sorted(scores.items(), key=lambda x: x[0]):
    #     print(prepare_results(metric, results['p'], results['r'], results['f']))

    # plt.imshow(cosine)
    # plt.show()


        # Print log info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='allcontent_required_NotNull_sum_out/allcontent_required_NotNull_sum_out.ckpt' , help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='allcontent_required_NotNull_sum_out/allcontent_required_NotNull_sum_outCaptions.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--dictionary', type=str, default='allcontent_required_NotNull_sum_out/allcontent_required_NotNull_sum_out.dict', help='path to dictionary file')
    ##parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--caption_path', type=str, default='allcontent_required_NotNull_sum_out/test.csv', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=200, help='step size for saving trained models')
    parser.add_argument('--image_dir', type=str, default='png/' , help='tmp')
    # parser.add_argument('--emb_model', type=str, default='fasttext.model', help='path for embedding model')
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=2, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
