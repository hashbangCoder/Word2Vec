## Use an Embedding Layer and initialize the Embeddings with pretrained vectors. Dont train embeddings.
	wordWeights,vec_dict = getWeights()
	## Do This otherwise, loss will give NaN. Due to bad initialization while finetuning
	wordWeights[0] = np.zeros((300,))
	vocabSize = wordWeights.shape[0]
## Index 0 is reserved for <MASK> token. Word indexes start from 1
## Run file from directory
import numpy as np
import cPickle, os
import pandas as pd
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-v", "--vectors", dest="vec_dict",help="Word-to-Index Dictionary path",default = '../saved_files/word2Ind_weights_dict.wts')
parser.add_option("-p", "--pretrained", dest="pretrained",help="Weights file path",default = '../saved_files/wordEmbedMat.pkl')
parser.add_option("-o", "--output", dest="outputFile",help="Weights output file name",default = '../Output/LSTM_pretrained')
parser.add_option("-i", "--input", dest="input",help="Input CSV file path",default = '../Data/REF.csv')
parser.add_option("-s", "--split-ratio", dest="split",help="Train data percentage",default = 0.9)
parser.add_option("-l", "--vector-length", dest="veclen",help="Length of word embeddings",default = 300)
parser.add_option("-e", "--num-epochs", dest="nEpochs",help="Number of epochs",default = 15)
parser.add_option("--model_weights", dest="model_weights",help="Load model parameters from file",default = False)
parser.add_option("--add-info", dest="add_info",help="Additional Info you want stored with output",default = None)
parser.add_option("--gpu", dest="gpu",help="GPU device",default = 0)
(options, args) = parser.parse_args()

os.environ['THEANO_FLAGS'] = 'floatX=float32,lib.cnmem=1,device=gpu' + str(options.gpu)

import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import wordpunct_tokenize
import datetime,json
from keras.callbacks import ModelCheckpoint


def getWeights():
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

del Data 
wordWeights,vec_dict = getWeights()
vocabSize = wordWeights.shape[0]

model = Sequential()
model.add(keras.layers.embeddings.Embedding(vocabSize,options.veclen,weights=(wordWeights,),mask_zero = True,trainable = False))
model.add(keras.layers.recurrent.GRU(128,init='glorot_uniform',inner_init='orthogonal',activation='tanh',inner_activation='hard_sigmoid',return_sequences = True))
model.add(keras.layers.recurrent.GRU(64,init='glorot_uniform',inner_init='orthogonal',activation='tanh',inner_activation='hard_sigmoid'))
model.add(Dense(1,activation = 'sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

numEpochs = int(options.nEpochs)
checkpointer = ModelCheckpoint(filepath='../Output/ModelParams/LSTM_params_finetuned.h5', verbose=1, save_best_only=True)

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
Hist = model.fit(trainDataNumbers,trainLabels,batch_size = 64,nb_epoch = numEpochs,validation_data = (validDataNumbers,validLabels),verbose = 1i,callbacks = [checkpointer])
cur_time =  datetime.datetime.strftime(datetime.datetime.now(), '%dth-%H:%M:%S')
with open(options.outputFile+'_'+cur_time + '.json','w') as f:
	if options.add_info:
		Hist.history['info'] = options.add_info
	json.dump(Hist.history,f)
print 'Loss and Validations history stored in JSON format at  ',options.outputFile+'_'+cur_time
model.save_weights('../Output/ModelParams/LSTM_pretrained.h5')
print 'Model parameters stored at /Output/ModelParams/LSTM_pretrained.h5'

def precision_recall(validDataNumbers,ValidLabels,model):
        print '\n\nCalculating precision and recall for model...'
        predictions = model.predict(validDataNumbers)
        predictions =  (predictions > 0.5).astype('int32')
        c0_inds = np.where(validLabels == 0)[0]
        c1_inds = np.where(validLabels == 1)[0]
        truePos_0 = sum(predictions[c0_inds] == 0)
        truePos_1 = sum(predictions[c1_inds] == 1)

        c0_precision = truePos_0/float(sum(predictions == 0))
        c0_recall = truePos_0/float(len(c0_inds))
        c1_precision = truePos_1/float(sum(predictions == 1))
        c1_recall = truePos_1/float(len(c1_inds))

        print 'Class 1 (REF) \t---->\t\tPrecision: %f    \tRecall : %f' %(c1_precision,c1_recall)
        print 'Class 0 (NR) \t---->\t\tPrecision: %f    \tRecall : %f' %(c0_precision,c0_recall)
        return c1_precision,c1_recall,c0_precision,c0_recall


c1p,c1r,c0p,c0r = precision_recall(validDataNumbers,validLabels,model)

