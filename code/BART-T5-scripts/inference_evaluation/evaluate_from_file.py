import os
import sys
import json

#!pip install --no-cache-dir transformers==4.3.3 sentencepiece==0.1.95
#!pip install spacy==2.1.0
#!pip install torch
#!pip install datasets
#!pip install rouge_score
#!pip install pip install sentencepiece
#!pip install gitpython
#!pip install sacrebleu
#!pip install sentence_splitter==1.4
#!pip install neuralcoref==4.0.0
#!python -m spacy download en
#!python -m spacy validate
#!pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz

input_file = sys.argv[1]

out_ROUGE_name = f"{input_file}_ROUGE.json" #IMPORTANT: file to save individual ROUGE results to
out_BERTScore_name = f"{input_file}_BERTScores.json" #IMPORTANT: file to save individual BERTScore results to
#out_METEOR_name = f"{base_folder}/test_generated_{model_string}_cp{checkpoint}_METEOR.json" #IMPORTANT: file to save individual METEOR results to
out_diversity_name = f"{input_file}_diversity.json" #IMPORTANT: file to save individual length + diversity results to
out_stats_name = f"{input_file}_stats.txt" #IMPORTANT: file to save overall average results to


# load data from json file with generations. format: dictionary with keys ['document', 'edited_document', 'generated', 'id', 'summary'], and values as corresponding lists
def load_generations_from_json(filename):
    with open(filename) as f:
        loaded_json = json.loads(f.read())
    print(type(loaded_json))
    generations = loaded_json['generated']
    gt_outputs = loaded_json['output']
    print('generations:', generations[:3])
    print('gt_outputs:', gt_outputs[:3])
    #print('generations:', generations[-10:])
    #print('gt_outputs:', gt_outputs[-10:])
    print(len(generations))
    print(len(gt_outputs))
    return generations, gt_outputs

'''
def load_generations_from_txt(gt_filename,gen_filename):
    f1 = open(gt_filename,'r')
    gt_outputs = [x.strip() for x in f1.readlines()]
    f2 = open(gen_filename,'r')
    generations = [x.strip() for x in f2.readlines()]
    print('generations:', generations[:10])
    print('gt_outputs:', gt_outputs[:10])
    print('generations:', generations[-10:])
    print('gt_outputs:', gt_outputs[-10:])
    print(len(generations))
    print(len(gt_outputs))
    return generations, gt_outputs
'''

generations, gt_outputs = load_generations_from_json(input_file)

"""# ROUGE Evaluation"""

from datasets import load_dataset, load_metric  
rouge_scorer = load_metric("rouge")

#modified : inserted cell to store separate ROUGE results per example

rouge_results_beam5_64 = rouge_scorer.compute(
    predictions=generations,
    references=gt_outputs,
    rouge_types=["rouge1", "rouge2", "rougeL"],
    use_agregator=False, use_stemmer=False,
)

# Individual ROUGE-1/2/L
print(len(rouge_results_beam5_64['rouge1']))
print(len(rouge_results_beam5_64['rouge2']))
print(len(rouge_results_beam5_64['rougeL']))
#print(rouge_results_beam5_64['rouge1'])
#print(rouge_results_beam5_64['rouge2'])
#print(rouge_results_beam5_64['rougeL'])

# modified : cell inserted to save individual ROUGE results to a file

ROUGE_dict = {}
ROUGE_dict["rouge1"] = [x.fmeasure for x in rouge_results_beam5_64['rouge1']]
ROUGE_dict["rouge2"] = [x.fmeasure for x in rouge_results_beam5_64['rouge2']]
ROUGE_dict["rougeL"] = [x.fmeasure for x in rouge_results_beam5_64['rougeL']]

#save individual ROUGE scores to file below. please uncomment the last three lines of this cell to save
#out_ROUGE_name = "split/metrics/valid_outputs_bartpretrained_0.8_split_full_ROUGE.json"

import json
with open(out_ROUGE_name,"w") as ROUGE_f:
    json.dump(ROUGE_dict,ROUGE_f,indent=4)
ROUGE_f.close()
print("ROUGE scores written to file")

#modified : calculate ROUGE based on outputs from current run of notebook OR from generations in a .json file (based on variable in previous cell)

rouge_results_beam5_64 = rouge_scorer.compute(
    predictions=generations,
    references=gt_outputs,
    rouge_types=["rouge1", "rouge2", "rougeL"],
    use_agregator=True, use_stemmer=False,
)

# ROUGE-2/L
print(rouge_results_beam5_64['rouge1'])
print(rouge_results_beam5_64['rouge2'])
print(rouge_results_beam5_64['rougeL'])
print("Final average results: ", rouge_results_beam5_64['rouge1'].mid.fmeasure, rouge_results_beam5_64['rouge2'].mid.fmeasure, rouge_results_beam5_64['rougeL'].mid.fmeasure)

"""# BERTScore Evaluation"""

#modified : below cells are all inserted for purpose of BERTScore evaluation

#!pip install bert_score==0.3.8 #changed to specific version

import bert_score
from bert_score import BERTScorer

def create_scorer():
    # Create scorer object for passing to get_bert_score
    scorer = BERTScorer(lang="en", rescale_with_baseline=True, model_type='roberta-base')
    return scorer

def get_bert_score(hyp,ref,scorer):
    # hyp: hypothesis ref: reference scorer: Already created BERT Score object
    # Returns F1: BERT-Score F1 between hypothesis and reference
    # Note: Some settings need to be done while creating the scorer object e.g whether to normalize by baseline or not, or which BERT model to use
    hyp = hyp.strip()
    ref = ref.strip()
    P, R, F1 = scorer.score([hyp,],[ref,])
    P = float(P.data.cpu().numpy())
    R = float(R.data.cpu().numpy())
    F1 = float(F1.data.cpu().numpy())
    return P, R, F1

from tqdm import tqdm
import numpy as np

def evaluate_bertscore(references, generations, scorer):
    all_results_P = []
    all_results_R = []
    all_results_F1 = []
    for ref, gen in tqdm(zip(references,generations)):
        P, R, F1 = get_bert_score(gen,ref,scorer)
        all_results_P.append(P)
        all_results_R.append(R)
        all_results_F1.append(F1)
    final_P = np.average(all_results_P)
    final_R = np.average(all_results_R)
    final_F1 = np.average(all_results_F1)
    return all_results_P, all_results_R, all_results_F1, final_P, final_R, final_F1

scorer = create_scorer()

# evaluate BERTScore on the validation set. either based on outputs from current run of notebook OR from generations in a .json file (based on variable in earlier cell)
# note: BERTScore values may differ *very slightly* depending on run of the notebook for the same data

all_results_P, all_results_R, all_results_F1, final_P, final_R, final_F1 = evaluate_bertscore(gt_outputs, generations, scorer)

print(final_P, final_R, final_F1) #around 2 minutes to run on full 11332 validation examples. final_F1 is the important score to look at

# save individual BERTScore results to a file

BERTScores_dict = {}
BERTScores_dict["P"] = all_results_P
BERTScores_dict["R"] = all_results_R
BERTScores_dict["F1"] = all_results_F1

#save individual BERTScores to file below. please uncomment the last three lines of this cell to save

import json
with open(out_BERTScore_name,"w") as BERTScore_f:
    json.dump(BERTScores_dict,BERTScore_f,indent=4)
BERTScore_f.close()
print("BERTScore results written to file")


"""# Diversity Metrics"""

#functions for length, TTR, UTR

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

#get length, TTR, and UTR of documents
document_val_len = [get_length(doc) for doc in generations]
document_val_TTR = [TTR_score(doc) for doc in generations]
document_val_UTR = [UTR_score(doc) for doc in generations]    

print("Average results: {} len, {} TTR, {} UTR".format(np.average(document_val_len),np.average(document_val_TTR),np.average(document_val_UTR)))
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

#save overall average results/stats to a file

with open(out_stats_name,"w") as stats_f:
    stats_f.write(str(rouge_results_beam5_64['rouge1'].mid.fmeasure) + ', ' + str(rouge_results_beam5_64['rouge2'].mid.fmeasure) + ', ' + str(rouge_results_beam5_64['rougeL'].mid.fmeasure) + '\n')
    stats_f.write(str(final_P) + ', ' + str(final_R) + ', ' + str(final_F1) + '\n')
    #stats_f.write(str(np.average(all_meteor_scores)) + '\n')
    stats_f.write("{} len, {} TTR, {} UTR".format(np.average(document_val_len),np.average(document_val_TTR),np.average(document_val_UTR)))
stats_f.close() 
print("Overall average results/stats written to file")

"""# Calculate Statistical Significance P-Values"""