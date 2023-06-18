#this script determines the average length (in words) of the outputs of a file (which is an input argument to the script), mainly just for statistics purposes (e.g. to write in the final paper)

import sys
import json
from string import punctuation
import re
import pkg_resources
import numpy as np
import nltk
nltk.download('punkt')
from nltk import word_tokenize
import numpy as np

input_file = sys.argv[1]

def get_length(document):
    word_lst = word_tokenize(document)
    return len(word_lst)

def get_word_count(input_file):
    input_f = open(input_file,'r',encoding='UTF-8')
    input_json = json.loads(input_f.read())
    print(len(input_json))
    
    inputs = [x['input'] for x in input_json]
    outputs = [x['output'] for x in input_json]
    
    len_lst = [get_length(doc) for doc in outputs]
    len_avg = np.average(len_lst)
    print(f"Average length of {input_file}: {len_avg}")

get_word_count(input_file)