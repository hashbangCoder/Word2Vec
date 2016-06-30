import numpy as np
from sklearn.preprocessing import normalize
import cPickle, time, sys
from optparse import OptionParser
from nltk.tokenize import wordpunct_tokenize
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing.sequence import make_sampling_table
from keras.layers.core import Reshape, Dense, Merge, Flatten, Activation
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from collections import Counter
from tokenized_text import tokenize
parser = OptionParser()
parser.add_option("-p", "--pretrained", dest="pretrained", help="Pretrained word vector file name",default = '../Pretrained_WordVecs/glove.6B.300d.txt')
parser.add_option("-d", "--dir", dest="dir",help="Directory to read from",default = None)			## '../Data/transcripts/'
parser.add_option("--exist", dest="exist",help="Path to tokenized and pickled word list",default = False)	## '../Data/saved_tokenized_text.txt'

parser.add_option("-e","--exclude",dest = 'exclude', help = "Comma separated files to exclude", default = None)
parser.add_option("-w","--window-length",dest = 'window', help = "Length of window", default = 5)
parser.add_option("-f","--frequency-threshold",dest = 'freq', help = "Minimum threshold for considering words", default = 2)
parser.add_option("-n","--neg-samples",dest = 'nsamples', help = "# of negative samples", default = 7)
parser.add_option("-s","--sample-factor",dest = 'samp_factor', help = "Higher the value, more sampling of frequent words", default = 1e-5)
parser.add_option("-l","--vector-length",dest = 'veclen', help = "Length of embedding", default = 300)
parser.add_option("--epochs",dest = 'epochs', help = "Number of epochs", default = 1)
(options, args) = parser.parse_args()

print '\n\n\n\n'
veclen = int(options.veclen)
def extractVecs():
## Extract pretrained vectors (as a starting point) from Downloaded CSV
## Pandas read_csv breaks while reading text file. Very buggy. Manually read each line with np.
		print 'Extracting pretrained word embeddings from file'
		t0 = time.clock()
		with open(options.pretrained,'r') as f:
				content = [item.rstrip().lower().split(' ') for item in f.readlines()]

		globalWordFile = np.asmatrix(content,dtype = str)
		globalWordTokens = globalWordFile[:,0].astype('str')
		globalWordVectors = globalWordFile[:,1:].astype(np.float)
		globalWordFile = None

		print time.clock() - t0, " seconds taken for loading pretrained embeddings"
		return globalWordTokens,globalWordVectors



def getPretrainedEmbeddings(wordDict,globalWordTokens,globalWordVectors):
	OOV_words = 0
	## If not using 300-dim word2vec, initialize with random weights instead
	if veclen != 300:
		print 'Pretrained embeddings not found for %d-dim vectors. Initializing randomly' %veclen
		return np.random.uniform(-1,1,(len(wordDict),veclen))

	weightMatrix  = np.empty(shape = (len(wordDict)+1,veclen))
	for word,value in wordDict.iteritems():
		indValue = np.where(globalWordTokens == word.lower())[0]
		if bool(indValue):
			weightMatrix[value,:] = globalWordVectors[indValue[0],:]
		else:
			#print   '"%s" does not appear in the gLoVe Dataset. Assigned random Word Vector' %word
			weightMatrix[value,:] = np.random.uniform(-1,1,size =veclen)
			OOV_words+=1
	print 'Total of %d words were not found in Pretrained Data and were assigned random WordVecs.' %OOV_words
	return wordDict, weightMatrix


def delInfrequentTokens(wordList):
	count= 0
	wordInd = 1
	wordDict = Counter(wordList)
	sampleMatrix = [1.0]
	samp_factor = float(options.samp_factor)
	for key,value in wordDict.items():
		if value < int(options.freq):
			wordDict.pop(key)
			wordList = [value for value in wordList if value != key]
			count +=1
		else:
			sampleMatrix.append(samp_factor/(wordDict[key]**0.5))
			wordDict[key] = wordInd
			wordInd +=1
	print '%d word tokens removed with threshold = %d and Total Token Count : %d' %(count,int(options.freq),wordInd)
	#assert np.array([bool(item>=options.freq) for item in wordDict.itervalues()]).all(), 'Re-check wordDict token frequency'
	return wordList, wordDict, np.asarray(sampleMatrix,dtype = np.float)



## Get all text files for training
excluded = options.exclude
if options.dir and options.exist:
	print 'Specify either a directory to read from OR existing tokenized word corpus'

elif options.dir and (not options.exist):
	print 'Reading all text files and tokenizing...'
	wordList = tokenize(options.dir,exclude_files = excluded)
	completeWordDict = Counter(wordList)
	wordList, wordDict, sampleMatrix = delInfrequentTokens(wordList)
	print 'Tokenized, deleted rare tokens and now dumping pickled tuple of (wordList,wordDict,sampleMatrix) to file...'
	with open('saved_tokenized_text.txt','wb') as f:
		cPickle.dump((wordList,wordDict,sampleMatrix),f,cPickle.HIGHEST_PROTOCOL)
elif options.exist and (not options.dir):
	print 'Loading tokenized word list from preexisting file...'
	with open(options.exist,'rb') as f:
		wordList,wordDict,sampleMatrix  = cPickle.load(f)
else:
	print 'Require either a directory to read files OR existing tokenized word corpus'
	sys.exit()
completeWordDict = Counter(wordList)

## Remove infrequent tokens and create a word dict
#wordList, wordDict, sampleMatrix = delInfrequentTokens(wordList)
globalWordTokens,globalWordVectors = extractVecs()
wordDict, wordWeights = getPretrainedEmbeddings(wordDict,globalWordTokens,globalWordVectors)
## Add some noise for context vectors since wordVecs != contextVecs
contextWeights = wordWeights + np.random.uniform()
del globalWordVectors, globalWordTokens

## Convert wordList to list of integers
vocab_size = len(sampleMatrix)
wordListInts = []
for ind,item in enumerate(wordList):
	wordListInts.append(wordDict[item])
del wordList
print 'Generating Skipgram pairs for training...will take a while...'

## TODO : Make samplingMatrix realistic for freq-sampling and remove next line
sampleMatrix = None
couples,labels = skipgrams(wordListInts, vocab_size, window_size=int(options.window), negative_samples=int(options.nsamples), shuffle=True,
							categorical=False, sampling_table=sampleMatrix)


## This part taken from @zachmayer
model_word = Sequential()
model_word.add(Embedding(vocab_size, veclen, input_length=1, weights = (wordWeights,)))
model_word.add(Reshape((1,veclen)))

model_context = Sequential()
model_context.add(Embedding(vocab_size,veclen, input_length=1, weights = (contextWeights,)))
model_context.add(Reshape((1,veclen,)))

model = Sequential()
model.add(Merge([model_word,model_context], mode='dot',dot_axes=2))
model.add(Flatten())
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='mse', optimizer='adam')

couples  = np.array(couples)
print 'NN model compiled...Start training on the text corpus'
histObj  = model.fit([couples[:,0],couples[:,1]],labels,batch_size=32,nb_epoch=int(options.epochs),verbose = 1, shuffle = False)

with open('wordEmbedMat.pkl','wb') as f:
	cPickle.dump(model.layers[0].layers[0].get_weights()[0],f,cPickle.HIGHEST_PROTOCOL)
with open('word2Ind_dict.pkl','wb') as f:
	cPickle.dump(wordDict,f,cPickle.HIGHEST_PROTOCOL)


## Ported from deprecated Keras example
def testWord2Vec(testWords,weights,num_display=3):
	##Generate inverse word mapping for easy lookup
	invWordDict = {v: k for k, v in wordDict.iteritems()}

	## Normalize the trained weights for cosine similarity
	trainedWeights = normalize(weights,norm = 'l2', axis = 1)
	for word in testWords:
		try:
			embedding = trainedWeights[wordDict[word],:]
			prox = np.argsort(np.dot(embedding,trainedWeights.transpose())/np.linalg.norm(embedding))[-num_display:].tolist()		
			prox.reverse()
			print 'Closest word vector (by cosine similarity) for %s : '%word, [invWordDict[item] for item in prox]
			
		except KeyError:
			print '"%s" not found in the Trained Word Embeddings. Skipping...'%word
			pass


