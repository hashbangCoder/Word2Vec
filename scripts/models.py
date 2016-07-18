import keras,os
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding, Input,merge
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences



def BiDi(input_shape,vocabSize,veclen,wordWeights,nLayers,nHidden,lr):
	assert len(nHidden) == nLayers, '#Neurons for each layer does not match #Layers'
	r_flag = True
	_Input = Input(shape = (input_shape,),dtype = 'int32')
	E = keras.layers.embeddings.Embedding(vocabSize,veclen,weights=(wordWeights,),mask_zero = True)(_Input)
	for ind in range(nLayers):
		if ind == (nLayers-1):
			r_flag = False
		fwd_layer = keras.layers.recurrent.GRU(nHidden[ind],init='glorot_uniform',inner_init='orthogonal',activation='tanh',inner_activation='hard_sigmoid',return_sequences = r_flag)(E)
		bkwd_layer = keras.layers.recurrent.GRU(nHidden[ind],init='glorot_uniform',inner_init='orthogonal',activation='tanh',inner_activation='hard_sigmoid',return_sequences = r_flag,go_backwards = True)(E)
		E = merge([fwd_layer,bkwd_layer],mode = 'ave')
		#nHidden/= 2

	Output = Dense(1,activation = 'sigmoid')(Dropout(0.5)(E))
	model = Model(input = _Input, output = Output)

	opt = keras.optimizers.Adam(lr)
	model.compile(loss='binary_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])
	return model
		  

