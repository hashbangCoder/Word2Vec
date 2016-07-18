import numpy as np
import cPickle
import pandas as pd
import sys,os,random,h5py
from nltk.tokenize import wordpunct_tokenize


def load_data(filePath='../Data/REF.csv',task = 'other',split = 0.8):
	if task == 'other':
	
		data = pd.read_csv(filePath,error_bad_lines=False,header=None)
		trainSamples = int(options.split*Data.shape[0])

		trainData = Data.ix[:trainSamples,1].as_matrix()
		trainLabels = Data.ix[:trainSamples,2].as_matrix()
		trainLabels[np.where(trainLabels == 'NR')[0]] = 0
		trainLabels[np.where(trainLabels == 'REF')[0]] = 1

		validData = Data.ix[trainSamples:,1].as_matrix()
		validLabels = Data.ix[trainSamples:,2].as_matrix()
		validLabels[np.where(validLabels == 'NR')[0]] = 0
		validLabels[np.where(validLabels == 'REF')[0]] = 1
		return trainData,trainLabels,validData,validLabels
	
	elif task == 'all' : 
		fileList = os.listdir('../Data/REF/Counselor/')
		data, labels  = [],[]
		random.shuffle(fileList)
		for fileName in fileList:
			with open('../Data/REF/Counselor/'+fileName) as f:
				data.append(f.read().strip().lower())
				if 'REF' in fileName:
					labels.append(1)
				else:
					labels.append(0)
		trainSamples = int(len(data)*split)
		trainData, trainLabels =  data[:trainSamples], labels[:trainSamples]
		validData, validLabels = data[trainSamples:], labels[trainSamples:]			

		return trainData,trainLabels,validData,validLabels
			
	else:
		print '\nInvalid data loading option'
		sys.exit(1)

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

def saveResults(filePath,metadata,scores):
	c1_precision = np.mean(scores['c1_precision'])
	c1_recall = np.mean(scores['c1_recall'])
	c0_precision = np.mean(scores['c0_precision'])
	c0_recall = np.mean(scores['c0_recall'])
	accuracy = np.mean(scores['accuracy'])
	c1_fscore = np.mean(scores['c1_fscore'])
	c0_fscore = np.mean(scores['c0_fscore'])
	with open(filePath,'a+') as f:
		print >> f, str(metadata)
		print >> f, 'Accuracy : %f' %accuracy 
		print >> f, 'Class 1 (REF)--> Precision: %f  \tRecall : %f  \tF-Score : %f' %(c1_precision,c1_recall,c1_fscore)
		print >> f, 'Class 0 (NR)--> Precision: %f  \tRecall : %f  \tf-Score : %f' %(c0_precision,c0_recall,c0_fscore)
		print >> f, '\n\n\n\n'
		print 'Results saved to file'
	
def analyze_false(validData,validLabels,model,vec_dict,weightsPath = ''):	
	model.load_weights(weightsPath)
	'Calculating precision and recall for best model...'
	predictions = model.predict(validData)
	predictions =  (predictions > 0.5).astype('int32')
	mean_length = np.zeros((validData.shape[0]))
	for ind,row in np.ndenumerate(validData):
	        mean_length[ind] = len(wordpunct_tokenize(row.lower().strip()))
	positive_indexes = np.squeeze(predictions) == validLabels
	mean_false_length = np.mean(mean_length[positive_indexes])	
	mean_length = np.logical_not(mean_length[positive_indexes])


