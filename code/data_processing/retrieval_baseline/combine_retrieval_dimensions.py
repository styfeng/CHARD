#this script combines retrieval-baseline data for the three separate dimensions into a combined .txt file. This combined output file can be found in "data/all_generation_txt_files/retrieval_baseline"

import sys
import json
import random
import csv
random.seed(42)


import sys
import json
import random
import csv
random.seed(42)


def separate_dimensions(input_txt):
    input_f = open(input_txt,encoding='UTF-8')
    input_lines = [x.strip() for x in input_f.readlines()]
    print(len(input_lines))
    risk_factor_indices = []
    treatment_indices = []
    prevention_indices = []
    for i,line in enumerate(input_lines):
        if 'risk factor' in line:
            risk_factor_indices.append(i)
        elif 'treatment' in line:
            treatment_indices.append(i)
        elif 'prevention' in line:
            prevention_indices.append(i)
    print(len(risk_factor_indices))
    print(len(treatment_indices))
    print(len(prevention_indices))
    risk_factor_inputs = [input_lines[i] for i in (risk_factor_indices)]
    treatment_inputs = [input_lines[i] for i in (treatment_indices)]
    prevention_inputs = [input_lines[i] for i in (prevention_indices)]
    return risk_factor_indices, treatment_indices, prevention_indices, risk_factor_inputs, treatment_inputs, prevention_inputs
    

def combine_retrieval_dimensions(risk_factor_file,treatment_file,prevention_file,output_file,risk_factor_indices,treatment_indices,prevention_indices):
    rf_f = open(risk_factor_file,encoding='UTF-8')
    rf_lines = [x.strip() for x in rf_f.readlines()]
    print(len(rf_lines))    
    treat_f = open(treatment_file,encoding='UTF-8')
    treat_lines = [x.strip() for x in treat_f.readlines()]
    print(len(treat_lines)) 
    prev_f = open(prevention_file,encoding='UTF-8')
    prev_lines = [x.strip() for x in prev_f.readlines()]
    print(len(prev_lines))
    total_lines = []
    rf_counter = 0
    treat_counter = 0
    prev_counter = 0
    for i in range(141):
        if i in risk_factor_indices:
            total_lines.append(rf_lines[rf_counter])
            rf_counter += 1
        elif i in treatment_indices:
            total_lines.append(treat_lines[treat_counter])
            treat_counter += 1
        elif i in prevention_indices:
            total_lines.append(prev_lines[prev_counter])
            prev_counter += 1  
    print(rf_counter)
    print(treat_counter)
    print(prev_counter)
    print(len(total_lines))
    with open(output_file,'w',encoding='UTF-8') as out_f:
        out_f.writelines('\n'.join(total_lines))
    out_f.close()
    print("Lines written to file")


risk_factor_indices, treatment_indices, prevention_indices, risk_factor_inputs, treatment_inputs, prevention_inputs = separate_dimensions('generation_txt_files/combined_test_HF_inputs.txt')

combine_retrieval_dimensions('retrieval_baseline/retrieval_risk-factor_results_postproc.txt','retrieval_baseline/retrieval_treatment_results_postproc.txt','retrieval_baseline/retrieval_prevention_results_postproc.txt','retrieval_baseline/retrieval_postproc_combined.txt',risk_factor_indices,treatment_indices,prevention_indices)