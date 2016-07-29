import numpy as np
import cPickle
import pandas as pd
import sys,os,random,h5py
from nltk.tokenize import wordpunct_tokenize
from sklearn.cross_validation import KFold

def load_data(label,filePath='../Data/REF.csv',task = 'all',split = 0.8,generate_vectors = False):
	if task == 'other':
		pass

	elif task == 'all' : 
		fileList = os.listdir('../Data/REF/Counselor/')
		data, labels  = [],[]
		random.shuffle(fileList)
		for fileName in fileList:
			with open('../Data/REF/Counselor/'+fileName) as f:
				data.append(f.read().strip().lower())
				if label in fileName:
					labels.append(1)
				else:
					labels.append(0)
		if generate_vectors:
			return data
		return np.array(data,dtype = object),np.array(labels)
	#	trainSamples = int(len(data)*split)
	#	trainData, trainLabels =  data[:trainSamples], labels[:trainSamples]
	#	validData, validLabels = data[trainSamples:], labels[trainSamples:]			
	#	if label in ['SEEK','AUTO','PWOP','NGI','NPWP','AF','CON']:
	#		c_samples = np.bincount(trainLabels)#len(np.where(trainLabels == 1 )[0])
	#		#c0_samples = len(np.where(trainLabels == 0 )[0])
	#		print c_samples
	#		c_weights = trainSamples/(2.0*c_samples)
	#		class_weights_dict = {0 :c_weights[0] , 1:c_weights[1] }
	#	else:
	#		class_weights_dict = None
	#	return np.array(trainData,dtype=object),np.array(trainLabels,dtype='int32'),np.array(validData,dtype=object),np.array(validLabels,dtype='int32'),class_weights_dict
	#		
	#else:
	#	print '\nInvalid data loading option'
	#	sys.exit(1)

def dataIter(data,labels,trainInd,validInd,label):
		
	trainData, trainLabels =  data[trainInd], labels[trainInd]
	validData, validLabels = data[validInd], labels[validInd]
	
	if label in ['SEEK','AUTO','PWOP','NGI','NPWP','AF','CON']:
			c_samples = np.bincount(trainLabels)#len(np.where(trainLabels == 1 )[0])
			#c0_samples = len(np.where(trainLabels == 0 )[0])
			print c_samples
			c_weights = trainData.shape[0]/(2.0*c_samples)
			class_weights_dict = {0 :c_weights[0] , 1:c_weights[1] }
	else:
			class_weights_dict = None
	return np.array(trainData,dtype=object),np.array(trainLabels,dtype='int32'),np.array(validData,dtype=object),np.array(validLabels,dtype='int32'),class_weights_dict
			





def precision_recall(validDataNumbers,validLabels,model,weightsPath = ''):
	model.load_weights(weightsPath)
	'Calculating precision and recall for best model...'
	predictions = model.predict(validDataNumbers)
	predictions =  (predictions > 0.5).astype('int32')
	c0_inds = np.where(validLabels == 0)[0]
	c1_inds = np.where(validLabels == 1)[0]
	truePos_0 = sum(predictions[c0_inds] == 0)
	truePos_1 = sum(predictions[c1_inds] == 1)

	accuracy = np.mean(np.squeeze(predictions) == validLabels)
	c0_precision = truePos_0/float(sum(predictions == 0))
	c0_recall = truePos_0/float(len(c0_inds))
	c0_fscore = 2*c0_precision*c0_recall/(c0_precision+c0_recall)
	c1_precision = truePos_1/float(sum(predictions == 1))
	c1_recall = truePos_1/float(len(c1_inds))
	c1_fscore = 2*c1_precision*c1_recall/(c1_precision+c1_recall)
	print  'Accuracy : %f' %accuracy 
	print  'Class 1 (REF)--> Precision: %f  \tRecall : %f  \tF-Score : %f' %(c1_precision,c1_recall,c1_fscore)
	print  'Class 0 (NR)--> Precision: %f  \tRecall : %f  \tf-Score : %f' %(c0_precision,c0_recall,c0_fscore)

	return c1_precision,c1_recall,c0_precision,c0_recall,accuracy,c1_fscore,c0_fscore


def analyze_false(validData,validDataNumbers,validLabels,model):	
	'Calculating precision and recall for best model...'
	predictions = np.squeeze((model.predict(validDataNumbers) > 0.5).astype('int32'))
	c1_inds = np.where(validLabels == 1)[0]
	pos_inds = np.where((predictions+validLabels) == 2)[0] #np.squeeze(predictions) == validLabels
	neg_inds = np.setdiff1d(c1_inds,pos_inds)
	seq_lengths = np.zeros((validData.shape[0]))
	for ind,row in np.ndenumerate(validData):
	        seq_lengths[ind] = len(wordpunct_tokenize(row.lower().strip()))	

	mean_true_length = np.mean(seq_lengths[pos_inds])	
	mean_false_length = np.mean(seq_lengths[neg_inds])
	
	return mean_false_length,mean_true_length


def saveResults(filePath,metadata,scores):
	c1_precision = np.mean(scores['c1_precision'])
	c1_recall = np.mean(scores['c1_recall'])
	c0_precision = np.mean(scores['c0_precision'])
	c0_recall = np.mean(scores['c0_recall'])
	accuracy = np.mean(scores['accuracy'])
	c1_fscore = np.mean(scores['c1_fscore'])
	c0_fscore = np.mean(scores['c0_fscore'])
	mfl = np.mean(scores['mean_false_length'])
	mtl = np.mean(scores['mean_true_length'])
	si = scores['samples_info']
	with open(filePath,'a+') as f:
		print >> f, str(metadata)
		print >> f, 'Accuracy : %f' %accuracy 
		print >> f, 'Class 1 (REF)--> Precision: %f  \tRecall : %f  \tF-Score : %f' %(c1_precision,c1_recall,c1_fscore)
		print >> f, 'Class 0 (NR)--> Precision: %f  \tRecall : %f  \tf-Score : %f' %(c0_precision,c0_recall,c0_fscore)

		print >> f, 'Mean False Length : %f \tMean True Length : %f' %(float(mfl),float(mtl))
		for ind in range(len(si)):
			print >> f, 'Run %d : \nTrain Samples : %d \tPostive Train Samples : %d\nValid Samples : %d\t Postive Valid Samples : %d' %(ind+1,si[ind][0],si[ind][1],si[ind][2],si[ind][3])
		print >> f, '\n\n\n\n'
		print 'Results saved to file'
	

