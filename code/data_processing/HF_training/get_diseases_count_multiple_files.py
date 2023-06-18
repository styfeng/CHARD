#this script obtains the number of unique conditions/diseases across several files, mainly just for statistics purposes (e.g. to write in the final paper when describing CHARDAT's dataset statistics)

import sys
import json

def get_diseases_count(in_file_train,in_file_val,in_file_test):
    f = open(in_file_train,'r',encoding='UTF-8')
    train_lines = [x.strip() for x in f.readlines()]
    f = open(in_file_val,'r',encoding='UTF-8')
    val_lines = [x.strip() for x in f.readlines()]
    f = open(in_file_test,'r',encoding='UTF-8')
    test_lines = [x.strip() for x in f.readlines()]
    lines = train_lines + val_lines + test_lines
    unique_conditions = []
    for line in lines:
        condition = line.split(' <sep> ')[1]
        if condition not in unique_conditions:
            unique_conditions.append(condition)
    print(f"Number of unique conditions: {len(unique_conditions)}")

get_diseases_count('prevention/prevention_train_txt.txt','prevention/prevention_val_txt.txt','prevention/prevention_test_txt.txt')
get_diseases_count('treatment/treatment_train_txt.txt','treatment/treatment_val_txt.txt','treatment/treatment_test_txt.txt')
get_diseases_count('risk-factor/risk-factor_train_txt.txt','risk-factor/risk-factor_val_txt.txt','risk-factor/risk-factor_test_txt.txt')

get_diseases_count('risk-factor/risk-factor_train_txt.txt','prevention/prevention_train_txt.txt','treatment/treatment_train_txt.txt')
get_diseases_count('risk-factor/risk-factor_val_txt.txt','prevention/prevention_val_txt.txt','treatment/treatment_val_txt.txt')
get_diseases_count('risk-factor/risk-factor_test_txt.txt','prevention/prevention_test_txt.txt','treatment/treatment_test_txt.txt')