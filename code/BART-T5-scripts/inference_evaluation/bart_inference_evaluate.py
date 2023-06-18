import os
import sys

"""# CHANGE THIS CELL BELOW"""

#CHANGE values in this cell appropriately
size = sys.argv[1] #CHANGE: IMPORTANT: either 'base' or 'large', base model is smaller and quicker to train and may be sufficient for your purposes. especially since T5-large is much much slower and compute heavy thane T5-base due to having to disable fp16
default_seed=int(sys.argv[2]) #CHANGE: IMPORTANT: seed1: 42, seed2: 24 (typically you train two seeds/versions of a model and take average, for your purposes probably enough to stick with one seed)
ENCODER_MAX_LENGTH = int(sys.argv[3]) #CHANGE: IMPORTANT: set this to reasonable value to ensure most input texts in your dataset can fit onto the encoder model. this is the number of tokens, but you can approximate it with the number of words in the input texts. reasonable values: 32, 64, 128 (powers of 2). 64 should be enough for most purposes...
DECODER_MAX_LENGTH = int(sys.argv[4]) #CHANGE: IMPORTANT: set this to reasonable value to ensure most output texts in your dataset can fit onto the decoder model. this is the number of tokens, but you can approximate it with the number of words in the output texts. reasonable values: 32, 64, 128 (powers of 2). 64 should be enough for most purposes...
model_type = sys.argv[5] #CHANGE: a string to describe the type of trained model, e.g. 'baseline' vs. 'top1'
val_json_filename = sys.argv[6]
lr = float(sys.argv[7]) #CHANGE: IMPORTANT: need to choose a good learning rate. some reasonable values to try are 5e-06, 1e-05, 2e-05, 3e-05, 5e-05
checkpoint = sys.argv[8]
base_folder = f"trained_BART_models/BART-{size}_{default_seed}_{model_type}_{lr}" #IMPORTANT: model folder that contains the checkpoint folders
input_batch_size = int(sys.argv[9]) #DEFAULT = 32
GPU = sys.argv[10]
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

if 'unseen' in val_json_filename:
    testset_type = 'unseen' 
elif 'seen' in val_json_filename:
    testset_type = 'seen'
else:
    testset_type = 'combined'
#testset_type = sys.argv[11] #e.g. full, unseen, seen

"""# Preliminary Things to Run"""

input_model = "{}/checkpoint-{}".format(base_folder,checkpoint) #full path to input model
model_string = f"BART-{size}_{default_seed}_{model_type}_{lr}" #string describing model and its parameters to be used in the variables below

#Note that likely the two most important evaluation metrics would be ROUGE and BERTScore (for both metrics, higher = better)
    #For BERTScore three values are outputted (the third value is called F1 and is the important one)
    #For ROUGE, three values (ROUGE1, ROUGE2, ROUGEL) are also outputted, and all three are important. However, ROUGE2 is typically seen as most important.
out_generations_filename_txt = f"{base_folder}/test-{testset_type}_generated_{model_string}_cp{checkpoint}_txt.txt" #IMPORTANT: .txt file to save generated summaries to
out_generations_filename = f"{base_folder}/test-{testset_type}_generated_{model_string}_cp{checkpoint}_json.json" #IMPORTANT: .json file to save generated summaries to
out_ROUGE_name = f"{base_folder}/test-{testset_type}_generated_{model_string}_cp{checkpoint}_ROUGE.json" #IMPORTANT: file to save individual ROUGE results to
out_BERTScore_name = f"{base_folder}/test-{testset_type}_generated_{model_string}_cp{checkpoint}_BERTScores.json"
out_diversity_name = f"{base_folder}/test-{testset_type}_generated_{model_string}_cp{checkpoint}_diversity.json"
out_PPL_name = f"{base_folder}/test-{testset_type}_generated_{model_string}_cp{checkpoint}_PPL.json"
#out_self_ROUGE_name = f"{base_folder}/test_generated_{model_string}_cp{checkpoint}_self-ROUGE.json"
out_stats_name = f"{base_folder}/test-{testset_type}_generated_{model_string}_cp{checkpoint}_stats.txt" #IMPORTANT: file to save overall average results to

import torch
from packaging import version
if version.parse(torch.__version__) < version.parse("1.6"):
    from .file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast


from datasets import load_dataset, load_metric                                               
#from torch.cuda.amp import autocast # need torch >=1.6
try:                          
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM   
except:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM                  
from transformers import Trainer, TrainingArguments
from transformers import BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration

import transformers
print(transformers.__version__)

bart_tokenizer = AutoTokenizer.from_pretrained(f"facebook/bart-{size}", use_fast=False)

"""# Load CommonGen Data into HuggingFace Dataset"""

import json

# load data from json file. format: list of dictionaries, each containing one example, where keys of each dict are "document", "edited_key" (specified as argument), "id", and "summary" (and possibly additional keys, these are just the mandatory ones)
def load_from_json(filename):
    with open(filename) as f:
        loaded_json = json.loads(f.read())
    #print(type(loaded_json))
    #print(loaded_json[0])
    inputs = [x['input'] for x in loaded_json]
    #print('input:', inputs[0])
    outputs = [x['output'] for x in loaded_json]
    #print('output:', outputs[0])
    return loaded_json

loaded_val_json = load_from_json(val_json_filename)

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
    return loaded_dict

converted_val_json = convert_json_format(loaded_val_json)

#modified : this cell has been inserted

from datasets import Dataset

#convert the converted_json to huggingface dataset format
def load_huggingface_dataset_from_dict(converted_json):
    #train_dataset = Dataset.from_dict(converted_json["train"])
    #val_dataset = Dataset.from_dict(converted_json["validation"])
    loaded_hf_dataset = Dataset.from_dict(converted_json)
    #print(type(loaded_hf_dataset))
    #print(loaded_hf_dataset)
    return loaded_hf_dataset

loaded_hf_dataset = load_huggingface_dataset_from_dict(converted_val_json)

print(loaded_hf_dataset[:2])

"""#Generation / Decoding"""

LANGUAGE = 'en_XX' #maybe variable not required (e.g. not in config) for non-FB models
#LANGUAGE = 'es_XX'
# Fairseq language codes for the mlsum languages are: es_XX, fr_XX, de_DE, ru_RU, tr_TR

BEAM_SIZE = 5 #CHANGED : was 2
DECODER_EARLY_STOPPING = True #CHANGED : added #Vestigial for Pegasus
DECODER_LENGTH_PENALTY = 0.6 #CHANGED : added
DECODER_MIN_LENGTH = 1 #CHANGED : added
NO_REPEAT_NGRAM_SIZE = 3
device = 'cuda'

import nltk; nltk.download('punkt')

#load the model from the specified folder:
bart_model = BartForConditionalGeneration.from_pretrained(input_model)

if version.parse(transformers.__version__) <= version.parse("4.1.1"):
    bart_model._keys_to_ignore_on_load_missing = None
    bart_model._keys_to_ignore_on_save = None

bart_model.to(device)

# Utility functions for generating text from the model.
#below 3 functions to create decoding time minibatches

def make_batch(texts, tokenizer,  device, src_lang=LANGUAGE):
    """ texts is the list of strings to use as input.
    LANGUAGE is the fairseq language code used (e.g., "es_XX", "fr_XX", "de_DE",
      "ru_RU", "tr_TR").
    """
    batch = tokenizer.prepare_seq2seq_batch(src_texts = texts, src_lang=src_lang, max_length=ENCODER_MAX_LENGTH, padding='max_length', return_tensors='pt',truncation=True)
    batch_features = dict([(k, v.to(device)) for k, v in batch.items()])
    return batch_features

#NOTE 250005 is the lang_id for Spanish.
#CHANGED : note that two functions below are changed to include additional arguments (e.g. min_length)
def generate(batch_features,
             model,
             tokenizer,
             early_stopping=DECODER_EARLY_STOPPING,
             length_penalty=DECODER_LENGTH_PENALTY,
             min_length=DECODER_MIN_LENGTH,
             max_length=DECODER_MAX_LENGTH,
             no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE, #added
             num_beams=BEAM_SIZE,
             src_lang=LANGUAGE):
    lang_id = tokenizer.encode(src_lang)[0]
    outputs = model.generate(**batch_features,
                decoder_start_token_id=lang_id, #for BART
                num_beams=num_beams,
                length_penalty = length_penalty,
                early_stopping = early_stopping,
                min_length = min_length,
                max_length=max_length,
                no_repeat_ngram_size = no_repeat_ngram_size #added
                )
    out = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return out

#this function is especially important, need to put the right field (e.g. input_key like "document" needs to be correct, 'document' is the default)
#modified: input_key is now an argument rather than hardcoded, will be set later when calling "generate_from_hf_batch" function
def generate_from_hf_data(batch,
                          tokenizer,
                          model,
                          device,
                          src_lang=LANGUAGE,
                          early_stopping=DECODER_EARLY_STOPPING,
                          length_penalty=DECODER_LENGTH_PENALTY,                         
                          min_length=DECODER_MIN_LENGTH,
                          max_length=DECODER_MAX_LENGTH,
                          no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE, #added
                          num_beams=BEAM_SIZE,
                          input_key='input'):
    #texts = batch['document']
    #texts = batch['edited_document']
    texts = batch[input_key]
    batch_features = make_batch(texts, tokenizer, device, src_lang=LANGUAGE)
    out = generate(batch_features,
                   model,
                   tokenizer,
                   early_stopping=early_stopping,
                   length_penalty=length_penalty,
                   min_length=min_length,
                   max_length=max_length,
                   no_repeat_ngram_size=no_repeat_ngram_size, #added
                   num_beams=num_beams,
                   src_lang=src_lang)
    return out

#time required: around 20 mins using V100 (or 35 mins using P100, 55 mins T4) for entire validation split using bartpretrained (11332 generations)

print(BEAM_SIZE)
valid_output_beam4 = None
in_key = 'input'

#takes around 2 minutes for full validation split using P100
valid_output_beam5_64 = loaded_hf_dataset.map(
        lambda batch: {'generated': generate_from_hf_data(batch,
            bart_tokenizer,
            bart_model,
            device,
            src_lang=LANGUAGE,
            early_stopping=DECODER_EARLY_STOPPING,
            length_penalty=DECODER_LENGTH_PENALTY,
            min_length=DECODER_MIN_LENGTH,
            max_length=DECODER_MAX_LENGTH,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            num_beams=BEAM_SIZE,
            input_key=in_key)
            },
            batched=True,
            batch_size=input_batch_size) #32 by default

#print(valid_output_beam5_64)
print(valid_output_beam5_64["input"][:2])
print(valid_output_beam5_64["output"][:2]) #comment if test set
print(valid_output_beam5_64["generated"][:2])
print(len(valid_output_beam5_64["generated"]))

#Save Output Generations to .json File
import json
with open(out_generations_filename,"w") as f:
   json.dump(valid_output_beam5_64[:],f,indent=4)
f.close()

#Save output generations to .txt file
with open(out_generations_filename_txt,'w') as f:
    f.write('\n'.join(valid_output_beam5_64["generated"][:len(valid_output_beam5_64["generated"])]))
f.close()

print("generated text written to files!")

"""#Evaluation"""

#modified : cell inserted. allows to either calculate metrics based on outputs from current run of notebook or to calculate from generations in a .json file

#below binary variable set to True if reading generations from file, otherwise False if just calculating based on outputs from current run of the notebook
calculate_from_file = False #CHANGE accordingly, probably leave as False
import json

# load data from json file with generations. format: dictionary with keys ['document', 'edited_document', 'generated', 'id', 'summary'], and values as corresponding lists
def load_generations_from_json(filename):
    with open(filename) as f:
        loaded_json = json.loads(f.read())
    print(type(loaded_json))
    generations = loaded_json['generated']
    gt_outputs = loaded_json['output']
    print('generations:', generations[:3])
    print('gt_outputs:', gt_outputs[:3])
    return generations, gt_outputs

if calculate_from_file is True:
    generations_filename = "" #CHANGE appropriately
    generations, gt_outputs = load_generations_from_json(generations_filename)

"""# ROUGE Evaluation"""

from datasets import load_dataset, load_metric  
rouge_scorer = load_metric("rouge")

#modified : inserted cell to store separate ROUGE results per example

if calculate_from_file is False:

    rouge_results_beam5_64 = rouge_scorer.compute(
        predictions=valid_output_beam5_64["generated"],
        references=valid_output_beam5_64["output"],
        rouge_types=["rouge1", "rouge2", "rougeL"],
        use_agregator=False, use_stemmer=False,
    )

else:
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

if calculate_from_file is False:

    rouge_results_beam5_64 = rouge_scorer.compute(
        predictions=valid_output_beam5_64["generated"],
        references=valid_output_beam5_64["output"],
        rouge_types=["rouge1", "rouge2", "rougeL"],
        use_agregator=True, use_stemmer=False,
    )

else:
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

if calculate_from_file is False:
    all_results_P, all_results_R, all_results_F1, final_P, final_R, final_F1 = evaluate_bertscore(valid_output_beam5_64["output"], valid_output_beam5_64["generated"], scorer)
else:
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
if calculate_from_file is False:
    document_val_len = [get_length(doc) for doc in valid_output_beam5_64["generated"]]
    document_val_TTR = [TTR_score(doc) for doc in tqdm(valid_output_beam5_64["generated"])]
    document_val_UTR = [UTR_score(doc) for doc in tqdm(valid_output_beam5_64["generated"])]
else:
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
#Average results: 9.777501244400199 len, 0.95752620015624 TTR, 0.9998340799573436 UTR

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
avg_PPL, ppl_results = evaluate_perplexity(valid_output_beam5_64["generated"])

print("Writing individual results to files")
with open(out_PPL_name,"w") as fout_ppl:
    fout_ppl.write('\n'.join([str(p) for p in ppl_results]))
fout_ppl.close()
print("Individual PPL numbers written to file")


#save overall average results/stats to a file

with open(out_stats_name,"w") as stats_f:
    stats_f.write(str(rouge_results_beam5_64['rouge1'].mid.fmeasure) + ', ' + str(rouge_results_beam5_64['rouge2'].mid.fmeasure) + ', ' + str(rouge_results_beam5_64['rougeL'].mid.fmeasure) + '\n')
    stats_f.write(str(final_P) + ', ' + str(final_R) + ', ' + str(final_F1) + '\n')
    #stats_f.write(str(np.average(all_meteor_scores)) + '\n')
    stats_f.write("{} len, {} TTR, {} UTR \n".format(np.average(document_val_len),np.average(document_val_TTR),np.average(document_val_UTR)))
    stats_f.write(str(avg_PPL))# + '\n')
    #stats_f.write(str(inter_rouge1_avg) + ', ' + str(inter_rouge2_avg) + ', ' + str(inter_rougeL_avg))
stats_f.close() 
print("Overall average results/stats written to file")