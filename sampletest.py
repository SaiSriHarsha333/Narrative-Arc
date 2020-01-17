import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
# from data_loader import get_loader
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

import pandas as pd

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    print("length", len(vocab))

    # Build data loader
    # data_loader = get_loader(args.image_dir, args.caption_path, vocab,
    #                             args.dictionary, args.batch_size,
    #                             shuffle=True, num_workers=args.num_workers)
    data = input("Enter Topic: ")
    # Build the models
    #encoder = EncoderCNN(args.embed_size).to(device)
    dictionary = pd.read_csv(args.dictionary, header=0,encoding = 'unicode_escape',error_bad_lines=False)
    dictionary = list(dictionary['keys'])

    decoder = DecoderRNN(len(dictionary), args.hidden_size, len(vocab), args.num_layers).to(device)

    decoder.load_state_dict(torch.load(args.model_path, map_location=device))
    decoder.eval()


    # Train the models
    # total_step = len(data_loader)
    # for epoch in range(args.num_epochs):
    # for i, (array, captions, lengths) in enumerate(data_loader):
    array = torch.zeros((len(dictionary)))
    for val in data.split():
        # Set mini-batch dataset
        array[dictionary.index(val)] = 1
        # print("In sample", array)
    array = (array, )
    array = torch.stack(array, 0)
    array = array.to(device)
    # print("After", array)
    #captions = captions.to(device)
    # targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

    # Forward, backward and optimize
    #features = encoder(images)
    outputs = decoder.sample(array)

    count = 0
    sentence = ''
    for i in range(len(outputs)):
        sampled_ids = outputs[i].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            count += 1
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = sentence.join(' ')
        sentence = sentence.join(sampled_caption)

        # Print out the image and the generated caption
    print (sentence)

    print(count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str, default='collections_all_science_outLstmTwoUni.ckpt' , help='path of saved models')
    parser.add_argument('--vocab_path', type=str, default='collections_all_science_out.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--dictionary', type=str, default='collections_all_science_out.dict', help='path to dictionary file')
    # parser.add_argument('--caption_path', type=str, default='data/testdata.csv', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--image_dir', type=str, default='png/' , help='tmp')

    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=2, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
