## Use a Embedding Layer and initialize the Word Vectors with Gensim vectors for every word in Corpus. Update the vectors with training.
from optparse import OptionParser
import cPickle,json,os, sys


parser = OptionParser()
parser.add_option("-v", "--vectors", dest="vec_dict",help="Word-to-Index Dictionary path",default = '../saved_files/word2Ind_dict.pkl')  #Dict with word-Index mappings for Embedded layer
parser.add_option("-p", "--pretrained", dest="pretrained",help="Weights file path",default = '../saved_files/wordEmbedMat.pkl')     # Embed layer weights
parser.add_option("-o", "--output", dest="outputFile",help="Output file name",default = '../Output/GRU_BiDi')           # Output file name for losses, accuracy info
parser.add_option("-i", "--input", dest="input",help="Input CSV file path",default = '../Data/REF.csv')
parser.add_option("-w", "--output-weights", dest="outputWeights",help="Weights output file name",default ='../Output/ModelParams/LSTM_params_BiDi.h5')

parser.add_option("-R", "--runs", dest="runs",help="Number of runs to average results over",default = 1)
parser.add_option("-r", "--learning-rate", dest="lr",help="Learning Rate",default = 0.0001)
parser.add_option("-s", "--split-ratio", dest="split",help="Train data percentage",default = 0.8)
parser.add_option("-l", "--vector-length", dest="veclen",help="Length of word embeddings",default = 300)
parser.add_option("--num-epochs", dest="nEpochs",help="Number of epochs",default = 10)
parser.add_option("--num-layers", dest="nLayers",help="Number of layers",default = 2)
parser.add_option("--model-weights", dest="model_weights",help="Load model parameters from file",default = False)
parser.add_option("--add-info", dest="add_info",help="Additional Info you want stored with output",default = None)			  
parser.add_option("--neurons", dest="neurons",help="Neurons in each layer",default = '128,64')
parser.add_option("--gpu", dest="gpu",help="GPU device",default = 0)
parser.add_option('--main-info-file',help = 'File where all metadata will be stored' ,dest ='main_info_file', default ='')  #'../Output/results/LSTM_Bidi_results.txt'
parser.add_option('--label',help = 'MITI label to classify' ,dest ='label', default =None)  #'../Output/results/LSTM_Bidi_results.txt'
(options, args) = parser.parse_args()
os.environ['THEANO_FLAGS'] = 'floatX=float32,lib.cnmem=1,device=gpu' + str(options.gpu)
(options, args) = parser.parse_args()


import keras
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding, Input,merge
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import wordpunct_tokenize
import datetime,utils
import models
from keras import backend as K
def getWeights():
	with open(options.pretrained,'rb') as f:
		weights = cPickle.load(f)
	with open(options.vec_dict,'rb') as f:
		word_dict = cPickle.load(f)
	return weights,word_dict	

#Data = pd.read_csv(options.input,error_bad_lines=False,header=None)
#trainSamples = int(options.split*Data.shape[0])
#trainData = Data.ix[:trainSamples,1].as_matrix()
#trainLabels = Data.ix[:trainSamples,2].as_matrix()
#trainLabels[np.where(trainLabels == 'NR')[0]] = 0
#trainLabels[np.where(trainLabels == 'REF')[0]] = 1
#validData = Data.ix[trainSamples:,1].as_matrix()
#validLabels = Data.ix[trainSamples:,2].as_matrix()
#validLabels[np.where(validLabels == 'NR')[0]] = 0
#validLabels[np.where(validLabels == 'REF')[0]] = 1
#Data = None


scores = {'c1_precision' : [],'c1_recall' : [], 'c0_precision' : [], 'c0_recall' :  [], 'accuracy' : [],'c1_fscore' : [],'c0_fscore' : [],'mean_false_length':[], 'mean_actual_length' : [],'samples:info':[]}
wordWeights,vec_dict = getWeights()
vocabSize = wordWeights.shape[0]
wordWeights[0] = np.zeros((300,))
assert options.label in ['Question','REF','SEEK','AUTO','PWOP','NGI','NPWP','AF','CON'], 'Invalid Label provided for classification'

for run in range(int(options.runs)):

	trainData,trainLabels,validData,validLabels,class_weights = utils.load_data(label = options.label,task = 'all',split = float(options.split))
	trainData, trainLabels = np.array(trainData,dtype=object),np.array(trainLabels,dtype='int32')
	validData,validLabels = np.array(validData,dtype=object),np.array(validLabels,dtype='int32')

	print('Converting sentences to number sequences and padding lengths...')
	trainDataNumbers = []
	validDataNumbers = []
	for row in trainData:
		trainDataNumbers.append([vec_dict[word] for word in wordpunct_tokenize(row.lower().strip())])
	for row in validData:
		validDataNumbers.append([vec_dict[word] for word in wordpunct_tokenize(row.lower().strip())])

	input_shape = max(max([len(item) for item in trainDataNumbers]),max([len(item) for item in validDataNumbers]))
	trainDataNumbers = pad_sequences(trainDataNumbers,maxlen = input_shape,padding='post',dtype=np.int32)
	validDataNumbers = pad_sequences(validDataNumbers,maxlen = input_shape,padding = 'post', dtype=np.int32)
	assert trainDataNumbers.shape[0] == trainData.shape[0]

	nHidden = [int(item) for item in options.neurons.split(',')]
	nLayers = int(options.nLayers)
	model = models.BiDi(input_shape,vocabSize,int(options.veclen),wordWeights,nLayers,nHidden,float(options.lr))

	#Input = Input(shape = (input_shape,),dtype = 'int32')
	#E = keras.layers.embeddings.Embedding(vocabSize,options.veclen,weights=(wordWeights,),mask_zero = True)(Input)
	#for ind in range(int(options.nLayers)):
	#	if ind == int(options.nLayers) - 1:
	#		r_flag = False
	#
	#	fwd_layer = keras.layers.recurrent.GRU(nHidden,init='glorot_uniform',inner_init='orthogonal',activation='tanh',inner_activation='hard_sigmoid',return_sequences = r_flag)(E)
	#	bkwd_layer = keras.layers.recurrent.GRU(nHidden,init='glorot_uniform',inner_init='orthogonal',activation='tanh',inner_activation='hard_sigmoid',return_sequences = r_flag,go_backwards = True)(E)
	#	E = merge([fwd_layer,bkwd_layer],mode = 'sum')
	#	nHidden/= 2
	#
	#Output = Dense(1,activation = 'sigmoid')(Dropout(0.5)(E))
	#model = Model(input = Input, output = Output)
	#model.compile(loss='binary_crossentropy',
	#			  optimizer='adam',
	#			  metrics=['accuracy'])
	pos_train_samples = len(np.where(trainLabels == 1 )[0])
	pos_valid_samples = len(np.where(validLabels == 1 )[0])

	print '\n'*10
	print 'Number of Train Samples : %d' %len(trainData), '\tNumber of "%s" sub-samples : %d' %(options.label,pos_train_samples)
	print 'Number of Test Samples : %d' %len(validData), '\tNumber of "%s" sub-samples : %d' %(options.label,pos_valid_samples)  


	print '\n'*10
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
		print('Start Training Model...')
		Hist = model.fit(trainDataNumbers,trainLabels,batch_size = 64,nb_epoch = numEpochs,validation_data = (validDataNumbers,validLabels),verbose = 1,callbacks=[checkpointer,lr_anneal],class_weight = class_weights)
		curr_time =  datetime.datetime.strftime(datetime.datetime.now(), '%dth-%H:%M:%S')
		with open(options.outputFile+'_'+curr_time + '.pkl','w') as f:
			json.dump(Hist.history,f)

	elif os.path.exists(options.model_weights):
		model.load_weights(options.model_weights)
		opt = keras.optimizers.Adam(float(options.lr))
		model.compile(loss = 'binary_crossentropy',optimizer = opt,metrics = ['accuracy'] )
		raise KeyboardInterrupt


	print 'Loss and Validations history stored in ',options.outputFile+'_'+curr_time
	#model.save_weights('../Output/ModelParams/LSTM_params_BiDi_%s.h5'%curr_time)
	print 'Best Model parameters stored at ../Output/ModelParams/LSTM_BiDi_%s.h5' %curr_time
	mfalseLength,mactualLength = utils.analyze_false(validData,validDataNumbers,validLabels,model)
	c1p,c1r,c0p,c0r,acc,c1f,c0f = utils.precision_recall(validDataNumbers,validLabels,model,weightsPath = options.outputWeights)
	print 'Run %d results :-' %(run+1)
	scores['c1_precision'].append(c1p)
	scores['c1_recall'].append(c1r)
	scores['c0_precision'].append(c0p)
	scores['c0_recall'].append(c0r)
	scores['accuracy'].append(acc)
	scores['c1_fscore'].append(c1f)
	scores['c0_fscore'].append(c0f)
	scores['mean_actual_length'].append(mactualLength)
	scores['mean_false_length'].append(mfalseLength)
	scores['sample_info'].append((len(trainData),pos_train_samples,len(validData),pos_valid_samples))

results_info = curr_time + '\tHyperParameters:- \nWord-Index Dictionary : %s \tWordvectors file : %s \tLearning Rate : %f \t split-ratio : %f \tEpochs : %d \tOutput Weights : %s \tResults File : %s\n Neurons : %s \nModel-Layers: %s\nResults averaged over %d runs' %(options.vec_dict, options.pretrained, float(options.lr), float(options.split), int(options.nEpochs),options.outputWeights, options.outputFile,str(options.neurons),str(options.nLayers),int(options.runs))
results += '\nLabel : %s' %options.label

utils.saveResults(filePath = options.main_info_file,metadata = results_info,scores = scores)
#
#def precision_recall(validDataNumbers,ValidLabels,model):
#	'Calculating precision and recall for model...'
#	predictions = model.predict(validDataNumbers)
#        predictions =  (predictions > 0.5).astype('int32')
#	c0_inds = np.where(validLabels == 0)[0]
#	c1_inds = np.where(validLabels == 1)[0]
#	truePos_0 = sum(predictions[c0_inds] == 0)
#	truePos_1 = sum(predictions[c1_inds] == 1)
#		
#	c0_precision = truePos_0/float(sum(predictions == 0))
#	c0_recall = truePos_0/float(len(c0_inds))
#	c1_precision = truePos_1/float(sum(predictions == 1))
#	c1_recall = truePos_1/float(len(c1_inds))
#
#	print 'Class 1 (REF) -->\tPrecision: %f    \tRecall : %f' %(c1_precision,c1_recall)
#	print 'Class 0 (NR) -->\tPrecision: %f    \tRecall : %f' %(c0_precision,c0_recall)
#	return c1_precision,c1_recall,c0_precision,c0_recall
#
#
#c1p,c1r,c0p,c0r = precision_recall(validDataNumbers,validLabels,model)
