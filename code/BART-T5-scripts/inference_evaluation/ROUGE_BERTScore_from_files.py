import os
import sys
from datasets import load_dataset, load_metric 
import nltk; nltk.download('punkt')                                   
import json

file1 = sys.argv[1]
file2 = sys.argv[2]


def load_from_file(txt_file):
    txt_f = open(txt_file,encoding='UTF-8')
    txt_lines = [x.strip() for x in txt_f.readlines()]
    print(len(txt_lines))
    return txt_lines

generations = load_from_file(file1)
gt_outputs = load_from_file(file2)


rouge_scorer = load_metric("rouge")

rouge_results_beam5_64 = rouge_scorer.compute(
    predictions=generations,
    references=gt_outputs,
    rouge_types=["rouge1", "rouge2", "rougeL"],
    use_agregator=True, use_stemmer=False)

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