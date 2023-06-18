#this file preprocesses for backtranslation by separating the initial and explanation portions of complete passages and writes them to two separate .txt files, such as the ones found under "data/backtranslation_data/inputs".

#sample command: python prepare_explanations_for_backtranslation.py HF_training_data/combined/combined_train_HF.json backtranslation_data/inputs/combined_train_initials.txt backtranslation_data/inputs/combined_train_explanations.txt


import json
import sys

input_json = sys.argv[1] #HF_training_data "combined" training .json file, e.g. "combined_train_HF.json"
out_file_1 = sys.argv[2] #contains the "initial" part of passages which will not be backtranslated
out_file_2 = sys.argv[3] #contains the "explanation" part of passages which will be backtranslated


def extract_explanations(input_file,out_file_1,out_file_2):

    with open(input_file,'r') as HF_f:
        input_json = json.loads(HF_f.read())
    output_lst = []
    for json_dict in input_json:
        output_lst.append(json_dict["output"])
    print(len(output_lst))
    print(output_lst[:3])

    #input_f = open(input_file,'r',encoding='UTF-8')
    #input_lines = [x.strip() for x in input_f.readlines()]
    #input_lines_split = [x.split(' <sep> ') for x in input_lines]
    #answers = [x[3] for x in input_lines_split]
    
    initials = []
    explanations = []
    for answer in output_lst:
        because_index = 9000
        since_index = 9000
        as_index = 9000
        words = [x.strip() for x in answer.split()]
        if "because" in answer:
            because_index = words.index('because') #finds index of FIRST occurrence of item
        if "since" in answer: 
            since_index = words.index('since')
        if " as " in answer:   
            as_index = words.index('as')
        min_index = min(because_index,since_index,as_index)
        initial_words = words[:min_index+1]
        initial = ' '.join(initial_words)
        initials.append(initial)
        explanation_words = words[min_index+1:]
        explanation = ' '.join(explanation_words)
        explanations.append(explanation)

    with open(out_file_1,'w',encoding='UTF-8') as out_f1:
        out_f1.writelines('\n'.join(initials))
    out_f1.close()
    with open(out_file_2,'w',encoding='UTF-8') as out_f2:
        out_f2.writelines('\n'.join(explanations))
    out_f2.close()    
    print("Initials and explanations written to .txt files")

extract_explanations(input_json,out_file_1,out_file_2)