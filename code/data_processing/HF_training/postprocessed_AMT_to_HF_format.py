#this script converts postprocessed AMT annotation files (that contain annotated/human-written explanations by annotators) into final HF training .json files, by inserting the human-written explanations into full passages of the templated formats, and writing these as associated outputs to corresponding inputs. The postprocessed AMT annotation files (inputs to this script) are in "data/CHARDAT", and the final HF training .json files (outputs to this script) are in "data/HF_training_data".

#sample command: python postprocessed_AMT_to_HF_format.py prevention/prevention_train_txt.txt ../HF_training_data/prevention/prevention_train_HF.json prevention

import sys
import json

postprocessed_file = sys.argv[1]
output_file = sys.argv[2]
file_type = sys.argv[3] #'prevention', 'treatment', or 'risk factor'

def convert_to_HF(postprocessed_file, output_file, file_type):
    input_f = open(postprocessed_file,'r',encoding='UTF-8')
    input_lines = [x.strip() for x in input_f.readlines()]
    input_lines_split = [x.split(' <sep> ') for x in input_lines]
    print(input_lines[:2])
    print(input_lines_split[:2])
    output_lst = []
    for line in input_lines_split:
        output_dict = {}
        condition = line[1]
        strategy = line[2]
        output_dict["input"] = f'A person with {condition} has a/an {strategy} {file_type} because/since/as {{explanation}}'
        output_dict["output"] = line[3]
        output_lst.append(output_dict)
    print(len(output_lst))
    print(output_lst[:2])
    with open(output_file,'w',encoding='UTF-8') as out_f:
        json.dump(output_lst,out_f,indent=4)
    out_f.close()
    print("lines written to json file")

convert_to_HF(postprocessed_file, output_file, file_type)