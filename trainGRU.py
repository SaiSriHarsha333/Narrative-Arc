import tensorflow as tf
tf.enable_eager_execution()
import torch

import numpy as np
import os
import pickle
from data_loader import get_loader
from build_vocab import Vocabulary
import pandas as pd
from modelGRU import Decoder
from torch.nn.utils.rnn import pack_padded_sequence
import argparse
import unicodedata
import re
import os
import time
import string



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab,
                                args.dictionary, args.batch_size,
                                shuffle=True, num_workers=args.num_workers)

    # Build the models
    #encoder = EncoderCNN(args.embed_size).to(device)
    dictionary = pd.read_csv(args.dictionary, header=0,encoding = 'unicode_escape',error_bad_lines=False)
    dictionary = list(dictionary['keys'])

    decoder = Decoder(len(vocab), len(dictionary), args.units, args.batch_size)

    # Loss and optimizer
    optimizer = tf.train.AdamOptimizer()


    def loss_function(real, pred):
        mask = 1 - np.equal(real, 0)
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
        return tf.reduce_mean(loss_)

    # Train the models
    total_step = len(data_loader)
    # for epoch in range(args.num_epochs):
    #     for i, (array, captions, lengths) in enumerate(data_loader):
    #
    #         # Set mini-batch dataset
    #         array = array.to(device)
    #         captions = captions.to(device)
    #         targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
    #
    #         # Forward, backward and optimize
    #         #features = encoder(images)
    #         outputs = decoder(array, captions, lengths)
    #         loss = criterion(outputs, targets)
    #         decoder.zero_grad()
    #         #encoder.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         # Print log info
    #         if i % args.log_step == 0:
    #             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
    #                   .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
    #
    #         # Save the model checkpoints
    #         if (i+1) % args.save_step == 0:
    #             torch.save(decoder.state_dict(), os.path.join(
    #                 args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     decoder=decoder)

    EPOCHS = 1

    for epoch in range(EPOCHS):
        start = time.time()

        # hidden = encoder.initialize_hidden_state()
        total_loss = 0

        vocab_ins = Vocabulary()

        # for (batch, (inp, targ)) in enumerate(dataset):
        for i, (array, captions, lengths) in enumerate(data_loader):
            loss = 0
            array = array.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            with tf.GradientTape() as tape:
                # enc_output, enc_hidden = encoder(captions, hidden)
                dec_hidden = decoder.initialize_hidden_state()

                dec_input = tf.expand_dims([vocab('<start>')] * args.batch_size, 1)
                print(targets.shape)

                # Teacher forcing - feeding the target as the next input
                for t in range(1, targets.shape[0]):
                    # passing enc_output to the decoder
                    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, array)

                    loss += loss_function(targets[:, t], predictions)

                    # using teacher forcing
                    dec_input = tf.expand_dims(targets[:, t], 1)

            batch_loss = (loss / int(targets.shape[1]))

            total_loss += batch_loss

            variables = decoder.variables

            gradients = tape.gradient(loss, variables)

            optimizer.apply_gradients(zip(gradients, variables))

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # saving (checkpoint) the model every epoch
        checkpoint.save(file_prefix = checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / N_BATCH))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='modelsMath/' , help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--dictionary', type=str, default='utilsoutDict.dict', help='path to dictionary file')
    ##parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--caption_path', type=str, default='captions.csv', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=200, help='step size for saving trained models')
    parser.add_argument('--image_dir', type=str, default='png/' , help='tmp')

    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--units', type=int , default=1, help='number of units in GRU')

    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
