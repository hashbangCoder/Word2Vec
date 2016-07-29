## Use an Embedding Layer and initialize the Embeddings with pretrained vectors. Dont train embeddings.
## Index 0 is reserved for <MASK> token. Word indexes start from 1
## Run file from directory

from optparse import OptionParser
import datetime,os,h5py
parser = OptionParser()
parser.add_option("-v", "--vectors", dest="vec_dict",help="Word-to-Index Dictionary path",default = '../saved_files/word2Ind_dict.pkl')  #Dict with word-Index mappings for Embedded layer
parser.add_option("-p", "--pretrained", dest="pretrained",help="Weights file path",default = '../saved_files/wordEmbedMat.pkl')		# Embed layer weights
parser.add_option("-o", "--output", dest="outputFile",help="output file name",default = '../Output/LSTM_finetune_op')			# Output file name for losses, accuracy info
parser.add_option("-w", "--output-weights", dest="outputWeights",help="Weights output file name",default = '../Output/ModelParams/LSTM_params_finetuned.h5')
parser.add_option("-i", "--input", dest="input",help="Input CSV file path",default = '../Data/REF.csv')	
parser.add_option("-s", "--split-ratio", dest="split",help="Train data percentage",default = 0.8)
parser.add_option("-l", "--vector-length", dest="veclen",help="Length of word embeddings",default = 300)

parser.add_option("-r", "--learning-rate", dest="lr",help="Learning Rate",default = 0.001)
parser.add_option("-e", "--num-epochs", dest="nEpochs",help="Number of epochs",default = 10)
parser.add_option("-R", "--runs", dest="runs",help="Number of runs to average results over",default = 1)
parser.add_option("--neurons", dest="neurons",help="Neurons in each layer",default = '128,64')

parser.add_option('--label',help = 'MITI label to classify' ,dest ='label', default =None)  #'../Output/results/LSTM_Bidi_results.txt'
parser.add_option("--model_weights", dest="model_weights",help="Load model parameters from file",default = False)
parser.add_option("--add-info", dest="add_info",help="Additional Info you want stored with output",default = None)
parser.add_option('--main-info-file',dest ='main_info_file',help="File where all metadata will be stored", default ='')  #'../Output/results/LSTM_finetuned_results_embedds.txt'
parser.add_option("--gpu", dest="gpu",help="GPU device",default = 0)
(options, args) = parser.parse_args()
os.environ['THEANO_FLAGS'] = 'floatX=float32,lib.cnmem=1,device=gpu' + str(options.gpu)


import keras
import numpy as np
import cPickle,json
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from nltk.tokenize import wordpunct_tokenize
import utils
from sklearn.cross_validation import KFold


def getWeights():
	with open(options.pretrained,'rb') as f:
		weights = cPickle.load(f)
	with open(options.vec_dict,'rb') as f:
		word_dict = cPickle.load(f)
	return weights,word_dict	


scores = {'c1_precision' : [],'c1_recall' : [], 'c0_precision' : [], 'c0_recall' :  [], 'accuracy' : [],'c1_fscore' : [],'c0_fscore' : [],'mean_false_length':[], 'mean_true_length' : [],'samples_info':[]}
wordWeights,vec_dict = getWeights()
## Do This otherwise, loss will give NaN. Due to bad initialization while finetuning
wordWeights[0] = np.zeros((300,))
vocabSize = wordWeights.shape[0]
assert options.label  in ['Question','REF','SEEK','AUTO','PWOP','NGI','NPWP','AF','CON'], 'Invalid Label provided for classification'

data,labels = utils.load_data(label=options.label, task = 'all',split = float(options.split))
assert(len(data) == len(labels))
indIter = iter(KFold(len(data),int(options.runs),shuffle = True))
for run in range(int(options.runs)):
        try:     
	#trainData,trainLabels,validData,validLabels,class_weights = utils.load_data(label=options.label, task = 'all',split = float(options.split), = run)
            trainInd,validInd = indIter.next()
        except StopIteration:
            print('Stop Iteration. Breaking...')
            break
	trainData,trainLabels,validData,validLabels,class_weights = utils.dataIter(data,labels,trainInd,validInd,label = options.label)

	pos_train_samples = len(np.where(trainLabels == 1 )[0])
	pos_valid_samples = len(np.where(validLabels == 1 )[0])

	print '\n'*10
	print 'Number of Train Samples : %d' %len(trainData), '\tNumber of "%s" sub-samples : %d' %(options.label,pos_train_samples)
	print 'Number of Test Samples : %d' %len(validData), '\tNumber of "%s" sub-samples : %d' %(options.label,pos_valid_samples)  

	model = Sequential()
	neurons = [int(item) for item in options.neurons.split(',')]
	model.add(keras.layers.embeddings.Embedding(vocabSize,options.veclen,weights=(wordWeights,),mask_zero = True,trainable = False))
	#model.add(keras.layers.recurrent.GRU(neurons[0],init='glorot_uniform',inner_init='orthogonal',activation='tanh',inner_activation='hard_sigmoid',return_sequences = True))
	#model.add(keras.layers.recurrent.GRU(neurons[1],init='glorot_uniform',inner_init='orthogonal',activation='tanh',inner_activation='hard_sigmoid'))
	model.add(keras.layers.recurrent.GRU(neurons[0],init='glorot_uniform',inner_init='orthogonal',activation='tanh',inner_activation='hard_sigmoid'))
	#model.add(keras.layers.recurrent.SimpleRNN(128,init='glorot_uniform',inner_init='orthogonal',activation='tanh',return_sequences = True))
	#model.add(keras.layers.recurrent.SimpleRNN(64,init='glorot_uniform',inner_init='orthogonal',activation='tanh',))
	model.add(Dropout(0.5))
	model.add(Dense(1,activation = 'sigmoid'))
	opt = keras.optimizers.Adam(float(options.lr))
	model.compile(loss='binary_crossentropy',
				  optimizer=opt,
				  metrics=['accuracy'])

	checkpointer = ModelCheckpoint(filepath=options.outputWeights, verbose=1, save_best_only=True,monitor='val_acc',mode = 'max')
	class lrAnneal(keras.callbacks.EarlyStopping):

		def __init__(self, monitor='val_loss', patience=0, verbose=0, mode='auto',anneal = True):
			super(lrAnneal, self).__init__(monitor = 'val_acc', patience = 0, verbose = 0, mode = 'max')
			self.anneal = anneal
			self.patience = patience
		def on_epoch_end(self, epoch, logs={}):
			current = logs.get(self.monitor)
			if current is None:
				warnings.warn('Early stopping requires %s available!' % (self.monitor), RuntimeWarning)

			if self.monitor_op(current, self.best):
				self.best = current
				self.wait = 0
			else:
				if (self.wait == self.patience - 1) and self.anneal:
						print 'Halving Learning Rate...'
						K.set_value(self.model.optimizer.lr, K.get_value(self.model.optimizer.lr)/2)

				elif self.wait >= self.patience:
					print('Epoch %d: early stopping' % (epoch))
					self.model.stop_training = True
				self.wait += 1

	lr_anneal = lrAnneal(monitor='val_acc', patience=3, verbose=0, mode='max', anneal = True )

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
		Hist = model.fit(trainDataNumbers,trainLabels,batch_size = 64,nb_epoch = numEpochs,validation_data = (validDataNumbers,validLabels),verbose = 1,callbacks=[checkpointer,lr_anneal],class_weight = class_weights)
		cur_time =	datetime.datetime.strftime(datetime.datetime.now(), '%dth-%H:%M:%S')
		with open(options.outputFile+'_'+cur_time + '.json','w') as f:
			if options.add_info:
				Hist.history['info'] = str(options.add_info)
			json.dump(Hist.history,f)
		print 'Loss and Validations history stored in ',options.outputFile+'_'+cur_time
		#model.save_weights('../Output/ModelParams/LSTM_params_finetuned_%s.h5'%cur_time)
		#print 'Model parameters stored at ../Output/ModelParams/LSTM_finetuned_%s.h5' %cur_time
	elif os.path.exists(options.model_weights):
		model.load_weights(options.model_weights)
		model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'] )

	c1p,c1r,c0p,c0r,acc,c1f,c0f = utils.precision_recall(validDataNumbers,validLabels,model,weightsPath = options.outputWeights)
	mfalseLength,mactualLength = utils.analyze_false(validData,validDataNumbers,validLabels,model)
	print 'Run %d results :-' %(run+1)
	scores['c1_precision'].append(c1p)
	scores['c1_recall'].append(c1r)
	scores['c0_precision'].append(c0p)
	scores['c0_recall'].append(c0r)
	scores['accuracy'].append(acc)
	scores['c1_fscore'].append(c1f)
	scores['c0_fscore'].append(c0f)
	scores['mean_true_length'].append(mactualLength)
	scores['mean_false_length'].append(mfalseLength)
	scores['samples_info'].append((len(trainData),pos_train_samples,len(validData),pos_valid_samples))

results_info = cur_time + '\tHyperParameters:- \nWord-Index Dictionary : %s \tWordvectors file : %s \tLearning Rate : %f \t split-ratio : %f \tEpochs : %d \tOutput Weights : %s \tResults File : %s\n Neurons : %s \nModel : %s\nResults averaged over %d runs' %(options.vec_dict, options.pretrained, float(options.lr), float(options.split), int(options.nEpochs),options.outputWeights, options.outputFile,str(options.neurons),';'.join([item.name for item in model.layers]),int(options.runs))
results_info += '\nLabel : %s' %options.label

utils.saveResults(filePath = options.main_info_file ,metadata = results_info,scores = scores)
