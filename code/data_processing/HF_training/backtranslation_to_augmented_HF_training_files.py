#this function produces augmented HF training files by combining the original HF training .json files with files that contain backtranslations. Multiple backtranslated files can be given as input (in a specific order), and the script will continually add backtranslations in that order to produce higher augmented versions, saving each to a different filename indicated with an "Ax" in its name (where A is the amount of augmentation, e.g. 2x means original data plus backtranslations from the first backtranslation file).

import json
import sys
import random
random.seed(42)


def extract_explanations(input_json,initials_file,backtranslated_files,output_file_name):

    inputs_lst = []
    with open(input_json,'r') as HF_f:
        input_json = json.loads(HF_f.read())
    print(len(input_json))
    for d in input_json:
        inputs_lst.append(d['input'])
    print(len(inputs_lst))
    initials_f = open(initials_file,'r',encoding='UTF-8')
    initials_lines = [x.strip() for x in initials_f.readlines()]
    print(len(initials_lines))
    
    bt_files = [open(bt_file,'r',encoding='UTF-8') for bt_file in backtranslated_files]
    backtranslated_lines = [[x.strip() for x in bt_f.readlines()] for bt_f in bt_files]
    print(len(backtranslated_lines))
    print(len(backtranslated_lines[0]))
    
    output_json_lst = []
    for bt_lines in backtranslated_lines:
        counter = 0
        for input,initial,bt in zip(inputs_lst,initials_lines,bt_lines):
            bt_lower = bt[0].lower() + bt[1:]
            bt_answer = initial + ' ' + bt_lower
            if counter < 1:
                print(bt_lower)
                print(bt_answer)
            counter += 1
            bt_dict = {}
            bt_dict['input'] = input
            bt_dict['output'] = bt_answer
            input_json.append(bt_dict)
        current_json = input_json.copy()
        output_json_lst.append(current_json)
        #print(len(output_json_lst))
        #print(len(input_json))
        #print(input_json[:2])
        #print(input_json[:-2])
    
    print(len(output_json_lst))
    random.seed(42)
    
    for x in output_json_lst:
        print('\n')
        print(len(x))
        print(x[0])
        #print(x[-1])
        random.shuffle(x)
        print(x[0])
        #print(x[-1])
        
    out_names = [output_file_name + f'_{i+2}x.json' for i in range(len(backtranslated_files))]
    for x,out_name in zip(output_json_lst,out_names):
        with open(out_name,'w',encoding='UTF-8') as out_f:
            json.dump(x,out_f,indent=4)
        out_f.close()
        print(f"Lines written to {out_name}")


input_json = 'HF_training_data/combined/combined_train_HF.json'
initials_file = 'backtranslation_data/inputs/combined_train_initials.txt'
#explanations_file = 'backtranslation_data/inputs/combined_train_explanations.txt'
'''
backtranslated_files = ['backtranslation_data/backtranslations/combined_train_explanations_0.9.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.9_v2.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.9_v3.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.9_v4.txt']
output_file_name = 'HF_training_data/augmented/best-temp0.9/combined_train_best-temp0.9'
'''
backtranslated_files = ['backtranslation_data/backtranslations/combined_train_explanations_0.9.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.9_v2.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.9_v3.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.9_v4.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.9_v5.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.9_v6.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.9_v7.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.9_v8.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.9_v9.txt']
output_file_name = 'HF_training_data/augmented/best-temp0.9/combined_train_best-temp0.9'
'''
backtranslated_files = ['backtranslation_data/backtranslations/combined_train_explanations_0.9.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.7.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.5.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.8.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.4.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.6.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.9_v2.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.7_v2.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.5_v2.txt']
output_file_name = 'HF_training_data/augmented/diff-temps-bart/combined_train_diff-temps-bart'
'''
'''
backtranslated_files = ['backtranslation_data/backtranslations/combined_train_explanations_0.8.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.4.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.6.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.5.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.9.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.7.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.8_v2.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.4_v2.txt',\
                        'backtranslation_data/backtranslations/combined_train_explanations_0.6_v2.txt']
output_file_name = 'HF_training_data/augmented/diff-temps-t5/combined_train_diff-temps-t5'
'''

extract_explanations(input_json,initials_file,backtranslated_files,output_file_name)