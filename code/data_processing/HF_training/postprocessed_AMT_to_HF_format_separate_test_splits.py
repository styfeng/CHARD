#this script converts postprocessed AMT annotation files (that contain annotated/human-written explanations by annotators) into final HF test .json files split into the seen and unseen halves, by inserting the human-written explanations into full passages of the templated formats, and writing these as associated outputs to corresponding inputs. The postprocessed AMT annotation files (inputs to this script) are in "data/CHARDAT", and these final HF test .json files (outputs to this script) are in "data/HF_training_data".

#sample command: python postprocessed_AMT_to_HF_format_separate_test_splits.py prevention/prevention_test_txt.txt ../HF_training_data/prevention/prevention_test-unseen_HF.json ../HF_training_data/prevention/prevention_test-seen_HF.json prevention

import sys
import json

postprocessed_file = sys.argv[1]
output_file_unseen = sys.argv[2]
output_file_seen = sys.argv[3]
file_type = sys.argv[4] #'prevention', 'treatment', or 'risk factor'

chosen_test_diseases = ['depression', 'Costochondritis', 'thyroidcancer', 'rheumatoid']

def convert_to_HF(postprocessed_file, output_file_unseen, output_file_seen, file_type):
    input_f = open(postprocessed_file,'r',encoding='UTF-8')
    input_lines = [x.strip() for x in input_f.readlines()]
    input_lines_split = [x.split(' <sep> ') for x in input_lines]
    print(len(input_lines))
    print(input_lines[:2])
    print(input_lines_split[:2])
    output_lst_unseen = []
    output_lst_seen = []
    for line in input_lines_split:
        output_dict = {}
        condition = line[1]
        strategy = line[2]
        output_dict["input"] = f'A person with {condition} has a/an {strategy} {file_type} because/since/as {{explanation}}'
        output_dict["output"] = line[3]
        if condition in chosen_test_diseases:
            output_lst_unseen.append(output_dict)
        else:
            output_lst_seen.append(output_dict)
    print(len(output_lst_unseen))
    print(output_lst_unseen[:2])
    print(len(output_lst_seen))
    print(output_lst_seen[:2])
    with open(output_file_unseen,'w',encoding='UTF-8') as out_f_unseen:
        json.dump(output_lst_unseen,out_f_unseen,indent=4)
    out_f_unseen.close()
    with open(output_file_seen,'w',encoding='UTF-8') as out_f_seen:
        json.dump(output_lst_seen,out_f_seen,indent=4)
    out_f_seen.close()
    print("lines written to json files")

convert_to_HF(postprocessed_file, output_file_unseen, output_file_seen, file_type)