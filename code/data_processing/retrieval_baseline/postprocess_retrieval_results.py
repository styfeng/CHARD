#this script converts retrieval-baseline explanations into the final retrieval-baseline examples which contain the first (or "initial") part of the templates. Outputs can be found in "data/all_generation_txt_files/retrieval_baseline"

import sys
import json
import random
import csv
random.seed(42)


def read_csv(fn):
    line_lst = []
    with open(fn,encoding='UTF-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                line_lst.append(row)
                line_count += 1
        print(f'Processed {line_count} lines.')
    csv_file.close()
    return line_lst


def full_sentences(csv_lines,retrieval_file,dimension,out_file):
    retrieval_f = open(retrieval_file,encoding='UTF-8')
    retrieval_lines = [x.strip() for x in retrieval_f.readlines()]
    output_lines = []
    for csv_l,re_l in zip(csv_lines,retrieval_lines):
        condition = csv_l[1]
        item = csv_l[2]
        lowercase_explanation = re_l[:1].lower() + re_l[1:]
        sentence = 'A person with ' + condition + ' has a/an ' + item + ' ' + dimension + ' because/since/as ' + lowercase_explanation
        output_lines.append(sentence)
    with open(out_file,'w',encoding='UTF-8') as out_f:
        out_f.writelines('\n'.join(output_lines))
    out_f.close()
    print("lines written to file")


line_lst = read_csv('../prevention_batch_CSV.csv')
full_sentences(line_lst,'retrieval_prevention_results.txt','prevention','retrieval_prevention_results_postproc.txt')

line_lst = read_csv('../treatment_batch_CSV.csv')
full_sentences(line_lst,'retrieval_treatment_results.txt','treatment','retrieval_treatment_results_postproc.txt')

line_lst = read_csv('../risk-factor_batch_CSV.csv')
full_sentences(line_lst,'retrieval_risk-factor_results.txt','risk factor','retrieval_risk-factor_results_postproc.txt')