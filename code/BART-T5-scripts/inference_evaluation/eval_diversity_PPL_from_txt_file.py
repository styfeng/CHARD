import os
import sys
                                             
#from torch.cuda.amp import autocast # need torch >=1.6

import json

input_file = sys.argv[1]

# load data from json file. format: list of dictionaries, each containing one example, where keys of each dict are "document", "edited_key" (specified as argument), "id", and "summary" (and possibly additional keys, these are just the mandatory ones)
def load_from_txt(filename):
    f = open(filename,encoding='UTF-8')
    outputs = [x.strip() for x in f.readlines()]
    return outputs

converted_outputs = load_from_txt(input_file)
print(len(converted_outputs))
'''
#modified : this cell has been inserted

#convert loaded json format to format required to load into huggingface dataset. edited_key argument same as earlier
def convert_json_format(loaded_json):#(loaded_train_json,loaded_val_json):
    loaded_dict = {}
    input_lst = [x['input'] for x in loaded_json]
    #print(len(input_lst))
    loaded_dict['input'] = input_lst
    #print(loaded_dict['input'][:3])
    output_lst = [x['output'] for x in loaded_json]
    #print(len(output_lst))
    loaded_dict['output'] = output_lst
    #print(loaded_dict['output'][:3])
    return loaded_dict['output']

converted_outputs = convert_json_format(loaded_val_json)
'''

import nltk; nltk.download('punkt')
from string import punctuation
import re
import pkg_resources
import numpy as np
import nltk
nltk.download('punkt')
from nltk import word_tokenize


def get_length(document):
    word_lst = word_tokenize(document)
    return len(word_lst)


#get type-token ratio of document
def TTR_score(document):
    word_lst = word_tokenize(document)
    clean_word_lst = []

    for word in word_lst:
        clean_word_lst.append(word)

    unique_word_lst = set(clean_word_lst)
    if len(clean_word_lst) == 0:
        TTR = 0
    else:
        TTR = len(unique_word_lst) / len(clean_word_lst)
    #print("Document: ", document, " / TTR: ", TTR)
    return TTR


#get unique-trigram ratio of document
def UTR_score(document):
    # returns the unique trigram fraction in this population.
    # Higher the unique trigram fraction, more the diversity
    unique_trigrams = set()
    total_trigrams = 0

    #for i,hyp_i in enumerate(hyp_population):
    document_words = document.strip().split()
    if len(document_words)>=3:
        total_trigrams += len(document_words)-2
        for j in range(len(document_words)-2):
            trigram = " ".join(document_words[j:j+2])
            unique_trigrams.add(trigram)

    unique_trigram_fraction = len(unique_trigrams)/(total_trigrams+1e-10)
    if total_trigrams == 0: unique_trigram_fraction = 0.0
    return unique_trigram_fraction

import time
from tqdm.auto import tqdm, trange

document_val_len = [get_length(doc) for doc in converted_outputs]
document_val_TTR = [TTR_score(doc) for doc in converted_outputs]
document_val_UTR = [UTR_score(doc) for doc in converted_outputs]  

print("Average results: {} len, {} TTR, {} UTR".format(np.average(document_val_len),np.average(document_val_TTR),np.average(document_val_UTR)))
'''
print(document_val_len[0:20])
print(min(document_val_len)) #3
print(max(document_val_len)) #16
#print(len(document_val_len))
print(document_val_TTR[0:20])
print(min(document_val_TTR)) #0.7272727272727273
#print(len(document_val_TTR))
print(document_val_UTR[0:20])
print(min(document_val_UTR)) #0.8888888888790123
#print(len(document_val_UTR))
#Average results: 9.777501244400199 len, 0.95752620015624 TTR, 0.9998340799573436 UTR
'''

'''
# save individual diversity results to a file

diversity_dict = {}
diversity_dict["length"] = document_val_len
diversity_dict["TTR"] = document_val_TTR
diversity_dict["UTR"] = document_val_UTR

import json
with open(out_diversity_name,"w") as diversity_f:
    json.dump(diversity_dict,diversity_f,indent=4)
diversity_f.close()
print("Diversity results written to file")
'''

#NEW (added 6-4-2022): Perplexity (PPL) using GPT-2
import sys
import argparse
import json
import glob
from pprint import pprint
import os
import scipy
import math
import numpy as np
import math
from math import log
from collections import defaultdict
from collections import OrderedDict
import torch
import time
from transformers import *
from tqdm import tqdm
import nltk
nltk.download('punkt')
from nltk import word_tokenize
from nltk import tokenize
import itertools

#use pretrained gpt-2 for ppl evaluation
print("calculating perplexity...")
lm_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
lm_model = GPT2LMHeadModel.from_pretrained('gpt2')
lm_model.eval()

'''
#generation_file contains the generated outputs from GPT-2 (lines)
def load_data(generation_file):
    print("Reading lines...")
    f = open(generation_file, 'r')
    lines = f.readlines()
    lines = [x.strip('\n').strip('\ufeff') for x in lines]
    print("Read in ",len(lines)," lines")
    #assert len(lines) == 1583 #inserted for new test set generations
    print("First 10 lines: ",lines[:10])
    return lines
'''

#get the perplexity of a given sentence or document
def ppl_score(sentence):
    input_ids = torch.tensor(lm_tokenizer.encode(sentence)).unsqueeze(0) 
    with torch.no_grad():
        outputs = lm_model(input_ids, labels=input_ids)
    return math.exp(outputs[0].item())
    
#main function that returns average perplexity (ppl) of text
def evaluate_perplexity(lines):
    ppl_results = []
    for line in tqdm(lines):
        ppl_results.append(ppl_score(line))
    final_ppl_result = np.average(ppl_results)
    return final_ppl_result, ppl_results

#overall_results = {}
#lines = load_data(generation_file)
avg_PPL, ppl_results = evaluate_perplexity(converted_outputs)
print(f"Average PPL: {avg_PPL}")