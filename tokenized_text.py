import numpy as np
from nltk.tokenize import wordpunct_tokenize
from collections import Counter
import os 

## Take a directory as input, read all files in it and tokenize the text corpus



def tokenize(directory,exclude_files):
	full_content = ''
	for _file in os.listdir(directory):
		#disp_count = 5
		if exclude_files  and (_file in exclude_files):
			continue
		with open(directory+_file,'r') as f:
			contents = f.readlines()
			for item in contents:
				try:
					sentence = item.split('\t')[1].strip()
					full_content += sentence
				except IndexError:
					continue
				# if np.random.binomial(1,0.1):

				# 	print sentence
				# 	time.sleep(2)				
				# 	disp_count -=1 
				# 	if not disp_count:
				# 		print '*'*100
				# 		break
						
				# else:
				# 	print '#'

	return wordpunct_tokenize(full_content.lower())