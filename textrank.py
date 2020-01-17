from collections import OrderedDict
import numpy as np
import spacy, string, nltk
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd
import gensim
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

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


df = pd.read_csv("allcontent_required.csv")
df = df.drop_duplicates(subset=['resource_id'], keep='first')
df = df[pd.notnull(df['description'])]
df['description'] = df['title'] + " " + df['description']
df=df['description']
# df1 = pd.read_csv("collections_all_science_out-temp_lda.csv")
# df1 = df1.drop_duplicates(subset=['filename'], keep='first')
# df1 = df1['description']
# df_row_reindex = pd.concat([df, df1], ignore_index=True)

all_keywords, values = unique_keywords(df)
unique_values = list(pd.unique(values))
np.savetxt("allcontent_required.dict", unique_values, delimiter="\n", fmt='%s')

save_file = []
for i, val in enumerate(df):
    if(len(all_keywords[i])!=0):
        st = ""
        for j in range(len(all_keywords[i])):
            st = st + str(all_keywords[i][j]) + " "# + str(all_keywords[5*i + 1]) + " " + str(all_keywords[5*i + 2]) + " " + str(all_keywords[5*i + 3]) + " " + str(all_keywords[5*i + 4])
        save_file.append([st, val])

save_file = pd.DataFrame(save_file, columns=['tk', 'val'])
save_file.to_csv("allcontent_required_captions.csv", header=True, index=False)
