import argparse
import pandas as pd
import numpy as np

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
from collections import OrderedDict
import spacy, string, nltk
from spacy.lang.en.stop_words import STOP_WORDS
import gensim
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nlp = spacy.load('en_core_web_sm')
stopwords = set(stopwords.words('english'))
punctuation = set(string.punctuation)
lemmatize = WordNetLemmatizer()
words = set(nltk.corpus.words.words())

class TextRank4Keyword():
    """Extract keywords from text"""

    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight


    def set_stopwords(self, stopwords):
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True

    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences

    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs

    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())

    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1

        # Get Symmeric matrix
        g = self.symmetrize(g)

        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm

        return g_norm


    def get_keywords(self, number=10):
        """Print top number keywords"""
        keywords = []
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            keywords.append(key)
            # print(key + ' - ' + str(value))
            if i > number:
                break
        return keywords

    def analyze(self, text,
                candidate_pos=['NOUN', 'PROPN'],
                window_size=4, lower=False, stopwords=list()):
        """Main function to analyze text"""

        # Set stop words
        self.set_stopwords(stopwords)

        # Pare text by spaCy
        doc = nlp(text)

        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower) # list of list of words

        # Build vocabulary
        vocab = self.get_vocab(sentences)

        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)

        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)

        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))

        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]

        self.node_weight = node_weight


def unique_keywords(df):
    all_keywords = []
    values = []
    for item in df:
        # gensim.utils.simple_preprocess(item, deacc=True)
        doc = nlp(item)
        b = []
        for tok in doc:
	           if tok.is_stop != True and tok.pos_ != 'SYM' and \
    	           tok.tag_ != 'PRP' and tok.tag_ != 'PRP$' and \
                   tok.tag_ != '_SP' and tok.pos_ != 'NUM' and \
                   tok.dep_ != 'aux' and tok.dep_ != 'prep' and \
                   tok.dep_ != 'det' and tok.dep_ != 'cc' and \
                   tok.lemma_ != 'frac' and len(tok) != 1 and \
                   tok.lemma_.lower() in words and \
                   tok.lemma_.lower() not in stopwords and \
                   tok.lemma_.lower() not in punctuation:
                        b.append(lemmatize.lemmatize(tok.lemma_.lower()))

        # print(b)
        # print(" ".join(b))
        tr4w = TextRank4Keyword()
        tr4w.analyze(" ".join(b), candidate_pos = ['NOUN', 'PROPN'], window_size=4, lower=False)
        keyword = tr4w.get_keywords(5)
        all_keywords.append(keyword)
        values = values + keyword
    return all_keywords, values

def genereate(args,key_words):
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
    # data = input("Enter Topic: ")
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
    for val in key_words:
        # Set mini-batch dataset
        if(val in dictionary):array[dictionary.index(val)] = 1
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
    return sentence
    # print(count)

def Distance(x1,y1,x2,y2):
	return ((x1-x2)**2 + (y1-y2)**2 ) ** 0.5


def generateKeywords(camp,resource_map,X,Y):
	camp["distance"] =  0

	for idx,row in cmap.iterrows():
		cmap.loc[idx,"distance"] = Distance(row["X"],row["Y"],X,Y)

	cmap.sort_values(by=['distance'],inplace = True)
	resource_ids = cmap.resource_id.values[0:6]
	des = []
	for i in resource_ids:
		if(len(resource_map[resource_map.resource_id == i].Summarization.values)!=0):
			des.append(resource_map[resource_map.resource_id == i].Summarization.values[0])

	all_keywords, values = unique_keywords(des)
	keys = list(pd.unique(values))
	print(keys)
	return keys


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--computency_map", type=str, required=True, help="Path to the computency map")
	parser.add_argument("--resource", type=str, required=True, help="Path to the resource csv")

	parser.add_argument("--input_X", type=float, required=True, help="X_Cooridnate")
	parser.add_argument("--input_Y", type=float, required=True, help="Y_Cooridnate")
	parser.add_argument('--model_path', type=str, default='allcontent_required_NotNull_sum_out/allcontent_required_NotNull_sum_out.ckpt' , help='path of saved models')
	parser.add_argument('--vocab_path', type=str, default='allcontent_required_NotNull_sum_out/allcontent_required_NotNull_sum_outCaptions.pkl', help='path for vocabulary wrapper')
	parser.add_argument('--dictionary', type=str, default='allcontent_required_NotNull_sum_out/allcontent_required_NotNull_sum_out.dict', help='path to dictionary file')
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

	cmap = pd.read_csv(args.computency_map)
	resource_map = pd.read_csv(args.resource)

	cmap.sort_values(by = ['topic_volume', 'doc_volume'],inplace = True)

	near_topics = generateKeywords(cmap,resource_map,args.input_X,args.input_Y)

	sen = genereate(args,near_topics)
