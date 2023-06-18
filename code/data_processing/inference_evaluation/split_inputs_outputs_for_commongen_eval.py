#this file splits training HF .json files (found under "data/HF_training_data" into separate input and output .txt files, e.g. for CommonGen evaluation purposes (which expects .txt files where each line of the file is either the input or generation for a single example), which can be found in "data/input_GT-output_txt_files"

import sys
import json
import random


def split_files(input_file):
    input_f = open(input_file,'r',encoding='UTF-8')
    input_json = json.loads(input_f.read())
    print(len(input_json))
    
    inputs = [x['input'] for x in input_json]
    outputs = [x['output'] for x in input_json]
    
    assert len(inputs) == len(outputs)
    assert len(input_json) == len(inputs)
    
    output_file_1 = input_file.strip('.json') + '_inputs.txt'
    output_file_2 = input_file.strip('.json') + '_outputs.txt'
    
    with open(output_file_1,'w') as out_f_1:
        out_f_1.writelines('\n'.join(inputs))
    out_f_1.close()
    
    with open(output_file_2,'w') as out_f_2:
        out_f_2.writelines('\n'.join(outputs))
    out_f_2.close()    
    
    print("Lines written to .txt files")
    

input_files = ['combined/combined_test_HF.json','combined/combined_test-seen_HF.json','combined/combined_test-unseen_HF.json',\
                'prevention/prevention_test_HF.json','prevention/prevention_test-seen_HF.json','prevention/prevention_test-unseen_HF.json',\
                'risk-factor/risk-factor_test_HF.json','risk-factor/risk-factor_test-seen_HF.json','risk-factor/risk-factor_test-unseen_HF.json',\
                'treatment/treatment_test_HF.json','treatment/treatment_test-seen_HF.json','treatment/treatment_test-unseen_HF.json']

for in_f in input_files:
    split_files(in_f)