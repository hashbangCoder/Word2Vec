import numpy as np
import pandas as pd
import cPickle 
import time
from optparse import OptionParser
from autocorrect import spell
from nltk.tokenize import wordpunct_tokenize
import utils

parser = OptionParser()
parser.add_option("-o", "--output", dest="outputFile",
                  help="Weights output file name",default = '../saved_files/word_dict_with_index_weights_tuple.wts')
parser.add_option("-p", "--pretrained", dest="pretrained",
                  help="Pretrained word vector file name",default = '../Pretrained_WordVecs/glove.6B.300d.txt')
parser.add_option("-i", "--input", dest="input",help="input corpus file name", default = '../Data/REF.csv')

(options, args) = parser.parse_args()
#if len(args) < 2:
#        parser.error("Insufficient arguments")

def extractVecs():
## Pandas read_csv breaks while reading text file. Very buggy. Manually read each line.
	t0 = time.clock()
	with open(options.pretrained,'r') as f:
	        content = [item.rstrip().lower().split(' ') for item in f.readlines()]

	globalWordFile = np.asmatrix(content,dtype = str)
	globalWordTokens = globalWordFile[:,0].astype('str')
	globalWordVectors = globalWordFile[:,1:].astype(np.float)
	globalWordFile = None
	
	print time.clock() - t0, " seconds taken for loading and slicing gLoVe Word Vectors"
	return globalWordTokens,globalWordVectors



trainDataset = utils.load_data(label = '',generate_vectors = True)
oneBigDataString = '\n'.join(trainDataset)
word_tokens = sorted(set(wordpunct_tokenize(oneBigDataString)))
vocabSize = len(word_tokens)


globalWordTokens, globalWordVectors = extractVecs()
## Get index of word (in corpus) in the GloVe vector file
word_vecs = {}
word_ind = 1
OOV_words = 0
t0 = time.clock()
for word in word_tokens:
	try:
		indValue = np.where(globalWordTokens == word.lower())[0]
		if bool(indValue) is True:
			 word_vecs[word] = [word_ind,globalWordVectors[indValue[0],:]]	        
		else:
			print   '"%s" does not appear in the gLoVe Dataset. Assigned random Word Vector' %word
			word_vecs[word] = [word_ind,np.random.uniform(-2,2,size = 300)]
			OOV_words+=1
		word_ind +=1
		
	except Exception as e:
		print word,'\t', indValue, type(word)
		print e.message
print time.clock() - t0, " taken to process the text corpus and assign word vectors"

t0 = time.clock()
with open(options.outputFile,'w') as f:
	cPickle.dump(word_vecs,f,protocol =2)
print time.clock() - t0, "seconds taken to store the word vectors"
