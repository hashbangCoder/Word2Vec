## Use a Embedding Layer and initialize the Word Vectors with Gensim vectors for every word in Corpus. Update the vectors with training.

import keras
import numpy as np
import cPickle
import pandas as pd
from optparse import OptionParser
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import wordpunct_tokenize
import datetime

parser = OptionParser()
parser.add_option("-v", "--vectors", dest="vec_dict",
                  help="Weights Dictionary file path",default = 'wordVecs')
parser.add_option("-o", "--output", dest="outputFile",
                  help="Weights output file name",default = 'vec_weights.wts')
parser.add_option("-i", "--input", dest="input",
                  help="Input CSV file path",default = 'REF.csv')
parser.add_option("-s", "--split-ratio", dest="split",
                  help="Train data percentage",default = 0.9)
parser.add_option("-l", "--vector-length", dest="veclen",
                  help="Length of word embeddings",default = 300)
parser.add_option("-e", "--num-epochs", dest="nEpochs",
                  help="Number of epochs",default = 15)


(options, args) = parser.parse_args()

def generateWeightMat():
	with open(options.vec_dict,'rb') as f:
		vec_dict = cPickle.load(f)
	wordWeights = np.zeros(shape =(len(vec_dict)+1,options.veclen),dtype = np.float)
	for _,val in vec_dict.iteritems():
		wordWeights[val[0]] = val[1]
	return wordWeights,vec_dict


## No shuffling of data. Will need to bucket minibatches as per length and pad accordingly instead of padding to the longest sequence length.
## TODO : Use truncated BPTT for longer sequences
## Base Imple - Pad all sequences to longest
Data = pd.read_csv(options.input,error_bad_lines=False,header=None)
trainSamples = int(options.split*Data.shape[0])

trainData = Data.ix[:trainSamples,1].as_matrix()
trainLabels = Data.ix[:trainSamples,2].as_matrix()
trainLabels[np.where(trainLabels == 'NR')[0]] = 0
trainLabels[np.where(trainLabels == 'REF')[0]] = 1

validData = Data.ix[trainSamples:,1].as_matrix()
validLabels = Data.ix[trainSamples:,2].as_matrix()
validLabels[np.where(validLabels == 'NR')[0]] = 0
validLabels[np.where(validLabels == 'REF')[0]] = 1

Data = None
wordWeights,vec_dict = generateWeightMat()
vocabSize = wordWeights.shape[0]

model = Sequential()
model.add(keras.layers.embeddings.Embedding(vocabSize,options.veclen,weights=(wordWeights,),mask_zero = True))
model.add(keras.layers.recurrent.GRU(128,init='glorot_uniform',inner_init='orthogonal',activation='tanh',inner_activation='hard_sigmoid',return_sequences = True))
model.add(keras.layers.recurrent.GRU(64,init='glorot_uniform',inner_init='orthogonal',activation='tanh',inner_activation='hard_sigmoid'))
model.add(Dense(1,activation = 'sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

numEpochs = options.nEpochs

print('Converting sentences to number sequences and padding lengths...')
trainDataNumbers = []
validDataNumbers = []
for row in trainData:
	trainDataNumbers.append([vec_dict[word][0] for word in wordpunct_tokenize(row)])
for row in validData:
	validDataNumbers.append([vec_dict[word][0] for word in wordpunct_tokenize(row)])

trainDataNumbers = pad_sequences(trainDataNumbers,padding='post',dtype=np.int32)
validDataNumbers = pad_sequences(validDataNumbers,padding = 'post', dtype=np.int32)
assert trainDataNumbers.shape[0] == trainData.shape[0]
#trainData = validData = None

print('Start Training Model...')
Hist = model.fit(trainDataNumbers,trainLabels,batch_size = 64,nb_epoch = numEpochs,validation_data = (validDataNumbers,validLabels),verbose = 1)
cur_time =  datetime.datetime.strftime(datetime.datetime.now(), '%dth-%H:%M:%S')
with open(options.output+'_'+curr_time + '.pkl','r') as f:
	f.write(Hist.history)


