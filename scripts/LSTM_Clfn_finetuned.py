## Use an Embedding Layer and initialize the Embeddings with pretrained vectors. Dont train embeddings.
## Index 0 is reserved for <MASK> token. Word indexes start from 1
## Run file from directory
import keras
import numpy as np
import cPickle,json
import pandas as pd
from optparse import OptionParser
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import wordpunct_tokenize
import datetime,os,h5py

parser = OptionParser()
parser.add_option("-v", "--vectors", dest="vec_dict",help="Word-to-Index Dictionary path",default = '../saved_files/word2Ind_dict.pkl')  #Dict with word-Index mappings for Embedded layer
parser.add_option("-p", "--pretrained", dest="pretrained",help="Weights file path",default = '../saved_files/wordEmbedMat.pkl')		# Embed layer weights
parser.add_option("-o", "--output", dest="outputFile",help="Weights output file name",default = '../OutputInfo/LSTM_finetune_op')			# Output file name for losses, accuracy info
parser.add_option("-i", "--input", dest="input",help="Input CSV file path",default = '../Data/REF.csv')	
parser.add_option("-s", "--split-ratio", dest="split",help="Train data percentage",default = 0.9)
parser.add_option("-l", "--vector-length", dest="veclen",help="Length of word embeddings",default = 300)
parser.add_option("-e", "--num-epochs", dest="nEpochs",help="Number of epochs",default = 15)
parser.add_option("--model_weights", dest="model_weights",help="Load model parameters from file",default = False)
parser.add_option("--add-info", dest="add_info",help="Additional Info you want stored with output",default = None)

(options, args) = parser.parse_args()

#def generateWeightMat():
#	with open(options.vec_dict,'rb') as f:
#		vec_dict = cPickle.load(f)
#	wordWeights = np.zeros(shape =(len(vec_dict)+1,options.veclen),dtype = np.float)
#	for _,val in vec_dict.iteritems():
#		wordWeights[val[0]] = val[1]
#	return wordWeights,vec_dict

def getWeights():
	with open(options.pretrained,'rb') as f:
		weights = cPickle.load(f)
	with open(options.vec_dict,'rb') as f:
		word_dict = cPickle.load(f)
	return weights,word_dict	


## No shuffling of data. Will need to bucket minibatches as per length and pad accordingly instead of padding to the longest sequence length.
## TODO : Use truncated BPTT for longer sequences
## Base Imple - Pad all sequences to longest

#with open(options.input,'r') as f:
#	lines = f.readlines()
#data = []
#labels = []
#for ind,item in enumerate(lines):
#	contents = item.split('"')
#	data.append(contents[1].strip())
#	labels.append(contents[2].strip(',').strip())
#	assert (labels[-1] == 'NR') or (labels[-1] == 'REF'), 'Invalid Label %s at %d' %(labels[-1],ind)
#Data = np.array(data)
#labels = np.array(labels)
#shuffle = numpy.random.permutation(len(a))
#Data = Data[shuffle]
#labels = labels[shuffle]

Data = pd.read_csv(options.input,error_bad_lines=False,header=None)
trainSamples = int(options.split*Data.shape[0])
  
trainData = Data.ix[:trainSamples,1].as_matrix()
trainLabels = Data.ix[:trainSamples,2].as_matrix()
#trainData = Data[:trainSamples]
#trainLabels = Data[:trainSamples]
trainLabels[np.where(trainLabels == 'NR')[0]] = 0
trainLabels[np.where(trainLabels == 'REF')[0]] = 1

validData = Data.ix[trainSamples:,1].as_matrix()
validLabels = Data.ix[trainSamples:,2].as_matrix()
#validData = Data[trainSamples:]
#validLabels = Data[trainSamples:]
validLabels[np.where(validLabels == 'NR')[0]] = 0
validLabels[np.where(validLabels == 'REF')[0]] = 1

del Data
wordWeights,vec_dict = getWeights()
## Do This otherwise, loss will give NaN. Due to bad initialization while finetuning
wordWeights[0] = np.zeros((300,))
vocabSize = wordWeights.shape[0]

print '\n'*10
print 'Number of Train Samples : %d' %len(trainData), '\tNumber of "REF" sub-samples : %d' %len(np.where(trainLabels == 1 )[0]) 
print 'Number of Test Samples : %d' %len(validData), '\tNumber of "REF" sub-samples : %d' %len(np.where(validLabels == 1)[0])  

model = Sequential()
model.add(keras.layers.embeddings.Embedding(vocabSize,options.veclen,weights=(wordWeights,),mask_zero = True,trainable = False))
#model.add(keras.layers.recurrent.GRU(128,init='glorot_uniform',inner_init='orthogonal',activation='tanh',inner_activation='hard_sigmoid',return_sequences = True))
#model.add(keras.layers.recurrent.GRU(64,init='glorot_uniform',inner_init='orthogonal',activation='tanh',inner_activation='hard_sigmoid'))
model.add(keras.layers.recurrent.SimpleRNN(128,init='glorot_uniform',inner_init='orthogonal',activation='tanh',return_sequences = True))
model.add(keras.layers.recurrent.SimpleRNN(64,init='glorot_uniform',inner_init='orthogonal',activation='tanh',))
model.add(Dropout(0.5))
model.add(Dense(1,activation = 'sigmoid'))
opt = keras.optimizers.Adam()
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

if not options.model_weights:
	numEpochs = int(options.nEpochs)

	print('Converting sentences to number sequences and padding lengths...')
	trainDataNumbers = []
	validDataNumbers = []
	for row in trainData:
		trainDataNumbers.append([vec_dict[word] for word in wordpunct_tokenize(row.lower().strip())])
	for row in validData:
		validDataNumbers.append([vec_dict[word] for word in wordpunct_tokenize(row.lower().strip())])

	trainDataNumbers = pad_sequences(trainDataNumbers,padding='post',dtype=np.int32)
	validDataNumbers = pad_sequences(validDataNumbers,padding = 'post', dtype=np.int32)
	assert trainDataNumbers.shape[0] == trainData.shape[0]
	#trainData = validData = None

	print('Start Training Model...')
	print str(options.add_info)
	Hist = model.fit(trainDataNumbers,trainLabels,batch_size = 64,nb_epoch = numEpochs,validation_data = (validDataNumbers,validLabels),verbose = 1)
	cur_time =  datetime.datetime.strftime(datetime.datetime.now(), '%dth-%H:%M:%S')
	with open(options.outputFile+'_'+cur_time + '.json','w') as f:
		if options.add_info:
			Hist.history['info'] = str(options.add_info)
		json.dump(Hist.history,f)
	print 'Loss and Validations history stored in ',options.outputFile+'_'+cur_time
	model.save_weights('../OutputInfo/ModelParams/LSTM_params_finetuned_%s.h5'%cur_time)
	print 'Model parameters stored at ../OutputInfo/ModelParams/LSTM_finetuned_%s.h5' %cur_time
elif os.path.exists(options.model_weights):
	model.load_weights(options.model_weights)
	model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'] )

def precision_recall(validDataNumbers,ValidLabels,model):
	'Calculating precision and recall for model...'
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

	print 'Class 1 (REF) -->\tPrecision: %f    \tRecall : %f' %(c1_precision,c1_recall)
	print 'Class 0 (NR) -->\tPrecision: %f    \tRecall : %f' %(c0_precision,c0_recall)
	return c1_precision,c1_recall,c0_precision,c0_recall


c1p,c1r,c0p,c0r = precision_recall(validDataNumbers,validLabels,model)
