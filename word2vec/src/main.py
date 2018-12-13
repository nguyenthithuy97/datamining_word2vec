from wordvector import WordVector
from windowmodel import WindowModel
import docload

import numpy as np
import sklearn.utils

def load():
	files = ['../data/adventures_of_sherlock_holmes.txt',
        	'../data/hound_of_the_baskervilles.txt',
        	'../data/sign_of_the_four.txt']
	word_array, dictionary, num_lines, num_words = docload.build_word_array(
    	files, vocab_size=50000, gutenberg=True)

	print('Document loaded and processed: {} lines, {} words.'
      	.format(num_lines, num_words))

	print('Building training set ...')
	x, y = WindowModel.build_training_set(word_array)

	# shuffle and split 10% validation data
	x_shuf, y_shuf = sklearn.utils.shuffle(x, y, random_state=0)
	split = round(x_shuf.shape[0]*0.9)
	x_val, y_val = (x_shuf[split:, :], y_shuf[split:, :])
	x_train, y_train = (x[:split, :], y[:split, :])

	print('Training set built.')
	graph_params = {'batch_size': 32,
	                'vocab_size': np.max(x)+1,
	                'embed_size': 64,
	                'hid_size': 64,
	                'neg_samples': 64,
	                'learn_rate': 0.01,
	                'momentum': 0.9,
	                'embed_noise': 0.1,
	                'hid_noise': 0.3,
	                'optimizer': 'Momentum'}
	model = WindowModel(graph_params)
	print('Model built. Vocab size = {}. Document length = {} words.'
	      .format(np.max(x)+1, len(word_array)))

	print('Training ...')
	results = model.train(x_train, y_train, x_val, y_val, epochs=120, verbose=False)

	word_vector_embed = WordVector(results['embed_weights'], dictionary)
	word_vector_nce = WordVector(results['nce_weights'], dictionary)

def get_common_word():	
	print(word_vector_embed.most_common(100))
if __name__ == '__main__':
	load()	
