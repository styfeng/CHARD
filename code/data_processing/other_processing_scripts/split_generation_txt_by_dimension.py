#this script splits .txt files that contain combined outputs (either ground-truth or from models) from all dimensions into separate .txt files that contain outputs per dimension, which may be used for several different purposes (e.g. individual dimension evaluation purposes)

import sys
import json
import random
import csv
random.seed(42)


#this function gets the indices of the lines that correspond to each dimension from 'combined_test_HF_inputs.txt'
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
    
risk_factor_indices, treatment_indices, prevention_indices, risk_factor_inputs, treatment_inputs, prevention_inputs = separate_dimensions('combined_test_HF_inputs.txt')


#this function uses the above indices to extract the individual dimension lines and writes them to three separate .txt files
def split_by_dimensions(input_txt,risk_factor_indices,treatment_indices,prevention_indices):
    input_f = open(input_txt,encoding='UTF-8')
    input_lines = [x.strip() for x in input_f.readlines()]
    print(len(input_lines))   
    rf_lines = [input_lines[i] for i in risk_factor_indices]
    treatment_lines = [input_lines[i] for i in treatment_indices]
    prevention_lines = [input_lines[i] for i in prevention_indices]
    rf_file = input_txt + '_risk-factor.txt'
    treatment_file = input_txt + '_treatment.txt'
    prevention_file = input_txt + '_prevention.txt'
    with open(rf_file,'w',encoding='UTF-8') as rf_f:
        rf_f.writelines('\n'.join(rf_lines))
    rf_f.close()
    with open(treatment_file,'w',encoding='UTF-8') as treat_f:
        treat_f.writelines('\n'.join(treatment_lines))
    treat_f.close()
    with open(prevention_file,'w',encoding='UTF-8') as prev_f:
        prev_f.writelines('\n'.join(prevention_lines))
    prev_f.close()
    print("separate dimension lines written to .txt files")
  
#input_txt = 'test-combined_generated_T5-large_1e-05_42_combined_2x_temp0.6_cp738_txt.txt'
input_txt = 'combined_test_HF_outputs.txt'
split_by_dimensions(input_txt,risk_factor_indices,treatment_indices,prevention_indices)