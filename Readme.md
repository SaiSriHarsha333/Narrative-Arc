# data_loader_fastText.py
Util file gives batch wise data to training file by taking in FastText trained model and train_data file. The keywords converted to 256 size vectors using FastText. This keywords can also be from out of list of dictionaty of Keywords. When their are more than one Keywords, all the vectors of keywords gets added then divided by no. of keywords. ALso each word in the description (caption) are converted to unique number from the vocabulary of FastText. Finally this file supplies keywords vector, array which contains unique number for each word in the decription and array which contains length of each decription. For all the files where you are using FastText, make sure you are using gensim version > 3.8.0.

# data_loader_embedding.py
Same thing as above, it also supplies description as list of string. This is used in validation/getting scores like Rouge, cosine similarity on Doc2Vec or Word2Vec for models with FastText.

# data_loader.py
This plain data_loader file takes train_data file, list of keywords and gives data batch wise to train file in batch size. It creates one hot encoded vector for all the keywords. It cannot work for keywords which are not present in the list of keywords.ALso each word in the description (caption) are converted to unique number from the vocabulary which is created by build_vocab. Finally it gives one hot encoding of keywords, array which contains unique number for each word in the decription and array which contains length of each decription.

# data_loaderDoc2Vec.py
Same as data_loader_embedding used for validation/getting scores like Rouge, cosine similarity on Doc2Vec or Word2Vec. The difference is that this should be used for plain LSTM without FastText.

# fasttext.py
File to train TastText model. This file was downloaded from colab notebook. Upload this file to colab to use this. Make sure you are using gensim version > 3.8.0. Only this supports generation of vectors for out of vocabulary.

# lossText.py
To test Plain LSTM without FastText. This takes in Trained Doc2Vec model, trained Lstm model, vocabulary built using build_vocab, dictionary which contains list of keywords, test csv file, other parameters of model like no. of layers etc... This file imports build_vocab, data_loaderDoc2Vec and losstestModel.

#gen_new_sum.py :
Generate a virtual document for a given point X,Y on the competency map.
		command to run it: python gen_new_sum.py --computency_map <coompetency map csv>  --resource <collection file> --input_X <point X> --input_Y <point Y>

#gen_new_sum.py :
Generate a virtual document for a given point X,Y on the competency map using trained LSTM and FastText model.
		command to run it: python gen_new_sum.py --computency_map <coompetency map csv>  --resource <collection file> --input_X <point X> --input_Y <point Y> --embed_model <filename>

# lossTextFastText.py
Same thing as lossText, but to test LSTM with FastText. This doesnot require vocabulary. This takes dictionary which contains list of keywords, test csv file, trained FastText model and other model parameters like no. of layers etc... This file imports build_vocab, data_loader_embedings, model_fastText.

# losstestModel.py
This file contains LSTM model architecture and used to generate predictions (text). This contains two types of generation, one where we can specify the no. of words to be generated and other generates 100 words by default.

# model_fastText.py
This file contains LSTM architecture. This takes trained FastText model weights to generate embeddings from unique number of each word in the vocabulary of FastText.

# model.py
Plain LSTM architecture and function to generate text with maximum of 100 words.

# sample.py
File to generate text from keywords. This takes trained LSTM model, vocabulary, dictionary and other LSTM mode parameters.

# sampletestFastText.py
File to generate text from keywords. This takes trained LSTM model, FastText model, dictionary and other LSTM mode parameters.

# textrank.py
Takes original csv file which contains many unrequired columns and generates keywords from description/summarizations. Creates csv file with two columns. First column contains keywords and second column contains description. Also creates dictioanary with list of all unique keywords.

# train.py
To train plain LSTM model. This takes folder to save trained model for each epoch, vocabulary, dictioanary with list of all unique keywords and captions. This file imports build_vocab, model and data_loader.

# trainGRU.py
Experimented but not fully complemented GRU model instead of LSTM.

# To train
With textrank: give original corpus to textrank.py.
Give the generateed csv(captions) file to build_vocab to generate vocabulary pickle dump file.
give the generated csv(captions) file, dictionary and vocabulary to train.py.
Same way for all the models with FastText.

# to generate samples
Give the trained model, dictionary, vocabulary to sampletest.text.
Use sampletestFastText for LSTM with FastText.
